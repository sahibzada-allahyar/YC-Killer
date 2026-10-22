/-
Millennium Harness Lean rung, kernel-side facts. Loads a compiled `Solution.olean` with `importModules` (its code is
never run: initializers stay disabled) and prints one JSON object. Python (millennium_harness/lean.py) makes every decision.

  lean --run Check.lean DIR              -- reference mode: fingerprint every user-facing constant of the module
  lean --run Check.lean DIR NAMES_FILE   -- check mode: fingerprint only the names listed (one per line)

Fingerprints are structural prints of the elaborated `Expr`, invariant to binder names, metadata, proofs (proof
irrelevance: every proof prints as ◾) and the names of auxiliary declarations (matchers etc. print as their body).
-/
import Lean
open Lean Meta

abbrev FpM := ReaderT ModuleIdx (StateRefT (NameMap String) MetaM)

def binder : BinderInfo → String
  | .implicit => "{}" | .strictImplicit => "⦃⦄" | .instImplicit => "[]" | .default => "()"

mutual
partial def canon (e : Expr) : FpM String := do
  if !e.isSort && !e.isLambda && !e.isForall then
    if (← try isProof e catch _ => pure false) then return "◾"
  match e with
  | .bvar i => return s!"#{i}"
  | .fvar id => return s!"x{(← id.getDecl).index}"
  | .mvar _ => return "?m"
  | .sort l => return s!"(Sort {l.normalize})"
  | .lit (.natVal n) => return toString n
  | .lit (.strVal s) => return s.quote
  | .mdata _ b => canon b
  | .proj s i b => return s!"({s}.{i} {← canon b})"
  | .const n ls => return s!"{← constStr n}.\{{ls.map Level.normalize}}"
  | .app .. => e.withApp fun f args => do
      let parts ← (#[f] ++ args).mapM canon
      return "(" ++ " ".intercalate parts.toList ++ ")"
  | .lam n t b bi => do
      let ts ← canon t
      withLocalDecl n bi t fun x => return s!"(λ{binder bi} {ts}, {← canon (b.instantiate1 x)})"
  | .forallE n t b bi => do
      let ts ← canon t
      withLocalDecl n bi t fun x => return s!"(∀{binder bi} {ts}, {← canon (b.instantiate1 x)})"
  | .letE n t v b _ => do
      let ts ← canon t
      let vs ← canon v
      withLetDecl n t v fun x => return s!"(let {ts} := {vs}; {← canon (b.instantiate1 x)})"

/-- Auxiliary constants of this module (matchers, `proof_n`, private/hygienic helpers) print as their body, so a
fingerprint does not depend on how many such helpers the agent created earlier in the file. -/
partial def constStr (n : Name) : FpM String := do
  let env ← getEnv
  unless n.isInternalDetail && env.getModuleIdxFor? n == some (← read) do return toString n
  if let some s := (← get).find? n then return s
  modify (·.insert n s!"<rec>")
  let s ← withLCtx {} {} do  -- closed terms, printed the same wherever they are first met
    match env.find? n with
    | some (.defnInfo v) => pure s!"<aux {← canon v.value}>"
    | some (.thmInfo v) => pure s!"<auxthm {← canon v.type}>"
    | some (.opaqueInfo v) => pure s!"<auxopaque {← canon v.value}>"
    | _ => pure (toString n)
  modify (·.insert n s)
  return s
end

def kindOf : ConstantInfo → String
  | .defnInfo _ => "def" | .thmInfo _ => "theorem" | .opaqueInfo _ => "opaque" | .axiomInfo _ => "axiom"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

/-- Namespaces whose appearance in a constant's type means meta-programming or IO happened in the module. -/
def metaPrefixes : List Name :=
  [`Lean.Elab, `Lean.Meta, `Lean.Macro, `Lean.MacroM, `Lean.Syntax, `Lean.ParserDescr, `Lean.TrailingParserDescr,
   `Lean.Parser, `Lean.Environment, `Lean.Kernel, `Lean.Core, `IO, `EIO, `BaseIO, `IO.RealWorld]

def fingerprint (mod : ModuleIdx) (n : Name) : FpM Json := do
  let env ← getEnv
  let some ci := env.find? n | return Json.mkObj [("exists", false)]
  let value? := match ci with
    | .defnInfo v => some v.value | .opaqueInfo v => some v.value | _ => none
  let ty ← canon ci.type
  let val ← match value? with | some v => do pure (Json.str (← canon v)) | none => pure Json.null
  let ctors := match ci with | .inductInfo v => v.ctors.map toString | _ => []
  return Json.mkObj [
    ("exists", true), ("kind", kindOf ci), ("type", ty), ("value", val), ("ctors", toJson ctors),
    ("type_sorry", ci.type.getUsedConstants.contains ``sorryAx),
    ("value_sorry", (value?.map (·.getUsedConstants.contains ``sorryAx)).getD false),
    ("axioms", toJson ((← collectAxioms n).map toString)),
    ("module", env.getModuleIdxFor? n == some mod),
    ("unsafe", ci.isUnsafe), ("partial", ci.isPartial),
    ("noncomputable", isNoncomputable env n),
    ("implemented_by", toJson ((Compiler.getImplementedBy? env n).map toString)),
    ("extern", isExtern env n),
    ("ir", (IR.findEnvDecl env n).isSome)]

/-- Solution-module inductive types that occur rigidly in the type (former) `e`. `e` and each argument walked into are
put in weak head normal form at `.all` transparency, so no reducibility attribute (global, scoped or `local`, which the
.olean does not record) can make an `abbrev`/`def` alias count as a new type. Only inductive-type and constructor heads
and arrows (and, with `lam`, the bodies of type families `fun _ => T`) are walked into; proofs, values, instance
arguments and stuck terms anchor nothing. -/
partial def anchors (mod : ModuleIdx) (e : Expr) (lam := false) : MetaM (Array Name) := do
  let e ← whnf e
  if let .forallE n d b bi := e then
    let a ← anchors mod d lam
    return a ++ (← withLocalDecl n bi d fun x => anchors mod (b.instantiate1 x) lam)
  if let .lam n d b bi := e then
    if lam then return ← withLocalDecl n bi d fun x => anchors mod (b.instantiate1 x) lam
  let .const c _ := e.getAppFn | return #[]
  let env ← getEnv
  let isInd := env.find? c matches some (.inductInfo _)
  unless isInd || env.find? c matches some (.ctorInfo _) do return #[]
  let mut out := if isInd && env.getModuleIdxFor? c == some mod then #[c] else #[]
  let info ← getFunInfoNArgs e.getAppFn e.getAppNumArgs
  for a in e.getAppArgs, i in [0:e.getAppNumArgs] do
    if (info.paramInfo[i]?.map (·.isInstImplicit)).getD false then continue
    if ← isTypeFormer a then out := out ++ (← anchors mod a lam)
  return out

/-- How many leading arguments of a coercion class describe the coerced value (its type, and for `CoeT`/`CoeDep` the
value itself); the remaining ones describe the target. 0 for any other class. -/
def coeSourceArity : Name → Nat
  | ``CoeT | ``CoeDep => 2
  | ``Coe | ``CoeTail | ``CoeOut | ``CoeHead | ``CoeTC | ``CoeOTC | ``CoeHTC | ``CoeHTCT | ``CoeFun | ``CoeSort => 1
  | _ => 0

/-- The Solution-module inductive type heading `e` after weak head normalisation (at the caller's transparency, `.all`
in `classFacts`, so an `abbrev`/`def` alias of it counts as it), if any. -/
def moduleHead (mod : ModuleIdx) (e : Expr) : MetaM (Array Name) := do
  let .const c _ := (← whnf e).getAppFn | return #[]
  let env ← getEnv
  return if env.find? c matches some (.inductInfo _) && env.getModuleIdxFor? c == some mod then #[c] else #[]

structure ClassFacts where
  cls : Name
  anchors : Array Name := #[]   -- input arguments (not `outParam`/`semiOutParam`/instance arguments)
  semi : Array Name := #[]      -- `semiOutParam` arguments (the source of `Coe`/`CoeTail`, the target of `CoeOut`...)
  mentions : Array Name := #[]  -- any argument but a coercion's source, type families included
  source : Array Name := #[]    -- the module's inductive type heading a coercion's source type, if one does: only a
                                -- value of that type can enter the coercion

/-- If a constant's type concludes, after unfolding every definition (an `abbrev MyRepr := Repr` is `Repr`), in an
application of a class `C`, return `C` and the Solution-module inductive types in its arguments. `anchors` are those
of its input arguments: type class resolution may leave `outParam`, `semiOutParam` and instance arguments as
metavariables, so an instance can only be found for a goal that mentions one of its anchors. -/
def classFacts (mod : ModuleIdx) (ty : Expr) : MetaM (Option ClassFacts) :=
  withTransparency .all <| forallTelescopeReducing ty fun _ body => do
    let body ← whnf body
    let .const cls _ := body.getAppFn | return none
    let env ← getEnv
    unless isClass env cls do return none
    let outs := (getOutParamPositions? env cls).getD #[]
    let mut binders : Array (BinderInfo × Bool × Bool) := #[]  -- binder info, `outParam`, `semiOutParam`
    let mut t := (← getConstInfo cls).type
    while true do
      let .forallE _ d b bi := t | break
      binders := binders.push (bi, d.isAppOfArity ``outParam 1, d.isAppOfArity ``semiOutParam 1)
      t := b
    let mut f : ClassFacts := { cls }
    for a in body.getAppArgs, i in [0:body.getAppNumArgs] do
      unless ← isTypeFormer a do continue
      let (bi, out, semi) := binders[i]?.getD (.default, false, false)
      if i ≥ coeSourceArity cls then f := { f with mentions := f.mentions ++ (← anchors mod a (lam := true)) }
      else if i == 0 then f := { f with source := (← moduleHead mod a) }
      if bi.isInstImplicit then continue
      if semi then f := { f with semi := f.semi ++ (← anchors mod a) }
      else if !(outs.contains i || out) then f := { f with anchors := f.anchors ++ (← anchors mod a) }
    return some f

/-- For a class-typed constant: its class, the classes of its instance arguments (and which of those are Prop-valued),
and whether it derives an instance for arbitrarily chosen types: a type variable of the conclusion (outside a
coercion's source) that no argument mentions which resolution has to find (an instance argument, or an argument the
goal does not determine, such as a constructor's field). With `deep` (the module's own constants) definitions are
unfolded as in `classFacts`; a library constant is read as written (it is only visited when its type syntactically
concludes in a class). -/
def graphFacts (ty : Expr) (deep : Bool) : MetaM (Option (Name × NameSet × NameSet × Bool)) :=
  withTransparency (if deep then .all else .reducible) <| forallTelescopeReducing ty fun xs body => do
    let env ← getEnv
    let body ← if deep then whnf body else pure body
    let .const cls _ := body.getAppFn | return none
    unless isClass env cls do return none
    let args := body.getAppArgs
    let src := coeSourceArity cls
    let mut hyps : NameSet := {}
    let mut props : NameSet := {}
    let mut blocking : Array Expr := #[]  -- types of the arguments resolution has to find
    for x in xs do
      let decl ← x.fvarId!.getDecl
      if !decl.binderInfo.isInstImplicit then
        unless body.containsFVar x.fvarId! do blocking := blocking.push decl.type
        continue
      blocking := blocking.push decl.type
      let h? ← forallTelescopeReducing decl.type fun _ b => do pure (← if deep then whnf b else pure b).getAppFn.constName?
      if let some h := h? then
        if isClass env h then
          hyps := hyps.insert h
          if ← isProp decl.type then props := props.insert h
    let mut isFree := false
    for x in xs do
      unless ← isTypeFormerType (← inferType x) do continue
      let occurs (e : Expr) := e.containsFVar x.fvarId!
      if args[:src].toArray.any occurs || !args[src:].toArray.any occurs then continue
      unless blocking.any occurs do isFree := true
    return some (cls, hyps, props, isFree)

/-- Type class plumbing of the whole environment (toolchain library, task and solution), for Python's closures:

* `feeds`: class `C` ↦ the classes of the instance arguments of constants whose type concludes in `C` (an instance
  `[D α] → C α` lets `D` instances decide what a `C` goal resolves to, or whether it resolves at all). Every
  class-typed constant counts, not only registered instances: `attribute [local instance]` can promote any of them,
  and the .olean does not record that. `prop_classes` lists the Prop-valued ones among the `D`s (a proof carries no
  data, but can still enable an instance).
* `free`: constants that derive an instance for arbitrarily chosen types (see `graphFacts`).
* `errors`: constants that could not be analysed. -/
def classGraph (mod : ModuleIdx) : MetaM Json := do
  let env ← getEnv
  -- The module's constants, and the library's whose type syntactically concludes in a class. Library proofs
  -- (instances of Prop-valued classes) only matter where an instance argument asks for their class: they are visited
  -- once such a class is known, until nothing new turns up.
  let mut todo : Array (Name × Expr × Bool) := #[]
  let mut proofs : Array (Name × Name × Expr) := #[]
  for (n, ci) in env.constants.toList do
    let inModule := env.getModuleIdxFor? n == some mod
    if inModule && !(ci matches .thmInfo _) then todo := todo.push (n, ci.type, true); continue
    let some head := ci.type.getForallBody.getAppFn.constName? | continue
    unless isClass env head do continue
    if inModule then todo := todo.push (n, ci.type, true)  -- a proof written as an instance
    else if ci matches .thmInfo _ then proofs := proofs.push (n, head, ci.type)
    else todo := todo.push (n, ci.type, false)
  let mut feeds : Std.HashMap Name NameSet := {}
  let mut props : NameSet := {}
  let mut free : Array Json := #[]
  let mut errors : Array Name := #[]
  while !todo.isEmpty do
    let mut next : Array (Name × Expr × Bool) := #[]
    for (n, ty, deep) in todo do
      let r? ← try some <$> graphFacts ty deep catch _ => pure none
      match r? with
      | none => errors := errors.push n
      | some none => pure ()
      | some (some (cls, hyps, ps, isFree)) =>
        unless hyps.isEmpty do feeds := feeds.insert cls (hyps.foldl (·.insert ·) (feeds.getD cls {}))
        for c in ps do
          unless props.contains c do
            props := props.insert c
            next := next ++ (proofs.filter (·.2.1 == c) |>.map fun (n, _, ty) => (n, ty, false))
        if isFree then free := free.push <| Json.mkObj [("name", toString n), ("class", toString cls)]
    todo := next
  let names (s : NameSet) := toJson (s.toList.map toString)
  return Json.mkObj [("feeds", Json.mkObj (feeds.toList.map fun (c, hs) => (toString c, names hs))),
    ("prop_classes", names props), ("free", toJson free), ("errors", toJson (errors.map toString))]

/-- Constants worth locking: user-facing names, not recursors or other auto-generated companions. -/
def userFacing (env : Environment) (ci : ConstantInfo) : Bool :=
  !ci.name.isInternalDetail && !(ci matches .recInfo _) && !isAuxRecursor env ci.name && !isNoConfusion env ci.name
    && !(ci.name.toString.startsWith "millennium_eof_")

def replayModule (md : ModuleData) : IO String := do
  try
    let base ← importModules md.imports {} (loadExts := false)
    let mut m : Std.HashMap Name ConstantInfo := {}
    for ci in md.constants do m := m.insert ci.name ci
    discard <| base.replay m
    return "ok"
  catch e => return toString e

def run (dir : System.FilePath) (names? : Option (Array Name)) : IO Json := do
  let sysroot := (← IO.appPath).parent.bind (·.parent) |>.getD "."
  initSearchPath sysroot
  searchPathRef.modify (· ++ [dir])  -- after the toolchain: a planted Init/Std .olean in DIR is never loaded
  let env ← importModules #[{ module := `Solution }] {} (loadExts := true)
  let some mod := env.getModuleIdx? `Solution | throw <| IO.userError "module Solution not loaded"
  let md := env.header.moduleData[mod.toNat]!
  let names := names?.getD <| md.constants.filter (userFacing env) |>.map (·.name)
  let ctx : Core.Context := { fileName := "<check>", fileMap := default, maxHeartbeats := 0 }
  let (fps, _) ← (do
      let mut out : Array (String × Json) := #[]
      for n in names do out := out.push (toString n, ← (fingerprint mod n).run mod |>.run' {})
      -- Every constant whose type is a class (instances, also local/scoped ones, which the .olean does not record
      -- as instances), with its class and the module's types in its arguments; "?" if it cannot be analysed (Python
      -- fails closed on it). Then the environment's class graph.
      let mut insts : Array Json := #[]
      for ci in md.constants do
        -- A theorem only when it is written as a class instance (one of a Prop-valued class can enable a library
        -- instance); unfolding an arbitrary statement could compute for as long as the statement likes.
        if ci matches .thmInfo _ && !(ci.type.getForallBody.getAppFn.constName?.map (isClass env)).getD false then
          continue
        let r ← try
            match ← classFacts mod ci.type with
            | some f => pure (some (toString f.cls, f, ""))
            | none => pure none
          catch e => pure (some ("?", { cls := .anonymous }, ← e.toMessageData.toString))
        if let some (cls, f, err) := r then
          let names (a : Array Name) := toJson (a.map toString)
          insts := insts.push <| Json.mkObj [("name", toString ci.name), ("class", cls), ("anchors", names f.anchors),
            ("semi", names f.semi), ("mentions", names f.mentions), ("source", names f.source), ("error", err)]
      return (out, insts, ← classGraph mod) : MetaM _).run' {} |>.toIO ctx { env }
  let (out, insts, graph) := fps
  let axiomsDeclared := md.constants.filter (·.isAxiom) |>.map (toString ·.name)
  let attrs := md.constants.filterMap fun ci =>
    if (Compiler.getImplementedBy? env ci.name).isSome || isExtern env ci.name then some (toString ci.name) else none
  let metaConsts := md.constants.filterMap fun ci =>
    if ci.type.getUsedConstants.any (fun c => metaPrefixes.any (·.isPrefixOf c)) then some (toString ci.name) else none
  -- `@[default_instance]` (always global, so recorded) decides the type of e.g. a numeral whose type is still open:
  -- every entry this module added, also one on a library instance (`attribute [default_instance] instOfNatFloat`).
  let defaultInsts := (defaultInstanceExtension.getModuleEntries env mod).map (toString ·.instanceName)
  return Json.mkObj [
    ("consts", Json.mkObj out.toList),
    ("module_consts", toJson (md.constNames.map toString)),
    ("imports", toJson (md.imports.map (toString ·.module))),
    ("axioms_declared", toJson axiomsDeclared), ("implemented_by_or_extern", toJson attrs),
    ("meta_consts", toJson metaConsts), ("instances", toJson insts), ("default_instances", toJson defaultInsts),
    ("class_graph", graph),
    ("replay", ← replayModule md)]

def main (args : List String) : IO UInt32 := do
  match args with
  | [dir] => IO.println (← run dir none).compress; return 0
  | [dir, namesFile] =>
      let names := (← IO.FS.lines namesFile).filter (· ≠ "") |>.map String.toName
      IO.println (← run dir (some names)).compress; return 0
  | _ => IO.eprintln "usage: lean --run Check.lean DIR [NAMES_FILE]"; return 2
