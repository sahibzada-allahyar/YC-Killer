/-
Two Lean FRO checkers on the pinned toolchain (Lean v4.23.0-rc2), for the Millennium Harness baselines
(bench/baselines/lean_baselines.py). Both print one JSON line.

  lean --run fro_tools_v423.lean comparator DIR CONFIG.json   -- DIR holds Challenge.olean and Solution.olean
  lean --run fro_tools_v423.lean lean4checker [--fresh] DIR MODULE

lean4checker (github.com/leanprover/lean4checker, tag v4.23.0-rc2): `lake exe lean4checker MODULE` = `replayFromImports`:
read MODULE's .olean, import its imports, and send every constant the module adds to the kernel with `replay'` (embedded
below verbatim). `--fresh` replays every constant, imported ones included, into an empty environment.

comparator (github.com/leanprover/comparator, HEAD ca04cfc, 2026-10-06, toolchain v4.35.0-rc4), ported to v4.23.0-rc2:

What is kept verbatim (Apache-2.0, Lean FRO; Authors: Henrik Böving): `Comparator/Compare.lean` (compareAt and its
worklist loop, definition holes), `Comparator/Axioms.lean` (checkAxioms), `Comparator/Util.lean` (runForUsedConsts),
`primitiveTargets`/`builtinTargets`, and `runBuiltinKernel` (replay of the exported closure into an empty environment,
then the Quot post-check). The config file has comparator's own schema (challenge_module, solution_module,
theorem_names, definition_names, permitted_axioms).

What changed, and why the decision is the same:
  * Export: comparator runs `lean4export MODULE -- TARGETS` and parses the NDJSON back into `constMap`. Here the same
    closure is computed in-process from the imported environment, following lean4export's `dumpConstant` rules (skip
    unsafe/partial constants, mdata stripped, an inductive pulls its whole mutual block + constructors + recursors, a
    ctor/recursor pulls its inductive, any Quot constant pulls Eq + the whole Quot package, a Nat literal pulls `Nat`,
    a string literal pulls `Char.ofNat` and the string literal constructor). A constant comparator would export but
    cannot find panics lean4export (LEAN_ABORT_ON_PANIC=1), i.e. a rejection; here it is the error `missing`.
  * Primitive targets: `String.ofList` and `eagerReduce` do not exist in v4.23 (string literals unfold to `String.mk`,
    which is already in the list); lean4export would abort on them for every solution, so they are dropped.
  * Kernel: v4.35's `Kernel.Environment.replay` (upstreamed from lean4checker) is lean4checker's `replay'` at its
    v4.23.0-rc2 tag, embedded below verbatim. Eq's closure is replayed first so the kernel's quotient initialisation
    finds `Eq` regardless of NameSet order (comparator's export always lists Eq before the Quot package).
  * No sandbox (landrun): it only isolates the build; it does not change accept/reject on these files.
Prints one JSON line: {"ok": bool, "stage": "...", "error": "..."}.
-/
import Lean
open Lean

/-! ### lean4checker v4.23.0-rc2, Lean4Checker/Replay.lean (verbatim; Apache-2.0, Kim Morrison)
`replay'` keeps a `Kernel.Environment` in its state. Core v4.23's `Environment.replay` is the same algorithm on an
`Environment`, whose `find?` does not see the auxiliary recursors of nested inductives (`Lean.Syntax.rec_1`) after a
replay, so it rejects every closure containing `Lean.Syntax` (comparator always exports `autoParam`, hence Syntax). -/
namespace Lean.Environment

namespace Replay'

structure Context where
  newConstants : Std.HashMap Name ConstantInfo

structure State where
  env : Kernel.Environment
  remaining : NameSet := {}
  pending : NameSet := {}
  postponedConstructors : NameSet := {}
  postponedRecursors : NameSet := {}

abbrev M := ReaderT Context <| StateRefT State IO

/-- Check if a `Name` still needs processing. If so, move it from `remaining` to `pending`. -/
def isTodo (name : Name) : M Bool := do
  let r := (← get).remaining
  if r.contains name then
    modify fun s => { s with remaining := s.remaining.erase name, pending := s.pending.insert name }
    return true
  else
    return false

/-- Use the current `Environment` to throw a `Kernel.Exception`. -/
def throwKernelException (ex : Kernel.Exception) : M Unit := do
  throw <| .userError <| (← ex.toMessageData {} |>.toString)

/-- Add a declaration, possibly throwing a `Kernel.Exception`. -/
def addDecl (d : Declaration) : M Unit := do
  match (← get).env.addDeclCore 0 d (cancelTk? := none) with
  | .ok env => modify fun s => { s with env := env }
  | .error ex => throwKernelException ex

mutual
/--
Check if a `Name` still needs to be processed (i.e. is in `remaining`).

If so, recursively replay any constants it refers to,
to ensure we add declarations in the right order.

The construct the `Declaration` from its stored `ConstantInfo`,
and add it to the environment.
-/
partial def replayConstant (name : Name) : M Unit := do
  if ← isTodo name then
    let some ci := (← read).newConstants[name]? | unreachable!
    replayConstants ci.getUsedConstantsAsSet
    -- Check that this name is still pending: a mutual block may have taken care of it.
    if (← get).pending.contains name then
      match ci with
      | .defnInfo   info =>
        addDecl (Declaration.defnDecl   info)
      | .thmInfo    info =>
        addDecl (Declaration.thmDecl    info)
      | .axiomInfo  info =>
        addDecl (Declaration.axiomDecl  info)
      | .opaqueInfo info =>
        addDecl (Declaration.opaqueDecl info)
      | .inductInfo info =>
        let lparams := info.levelParams
        let nparams := info.numParams
        let all ← info.all.mapM fun n => do pure <| ((← read).newConstants[n]!)
        for o in all do
          modify fun s =>
            { s with remaining := s.remaining.erase o.name, pending := s.pending.erase o.name }
        let ctorInfo ← all.mapM fun ci => do
          pure (ci, ← ci.inductiveVal!.ctors.mapM fun n => do
            pure ((← read).newConstants[n]!))
        -- Make sure we are really finished with the constructors.
        for (_, ctors) in ctorInfo do
          for ctor in ctors do
            replayConstants ctor.getUsedConstantsAsSet
        let types : List InductiveType := ctorInfo.map fun ⟨ci, ctors⟩ =>
          { name := ci.name
            type := ci.type
            ctors := ctors.map fun ci => { name := ci.name, type := ci.type } }
        addDecl (Declaration.inductDecl lparams nparams types false)
      -- We postpone checking constructors,
      -- and at the end make sure they are identical
      -- to the constructors generated when we replay the inductives.
      | .ctorInfo info =>
        modify fun s => { s with postponedConstructors := s.postponedConstructors.insert info.name }
      -- Similarly we postpone checking recursors.
      | .recInfo info =>
        modify fun s => { s with postponedRecursors := s.postponedRecursors.insert info.name }
      | .quotInfo _ =>
        addDecl (Declaration.quotDecl)
      modify fun s => { s with pending := s.pending.erase name }

/-- Replay a set of constants one at a time. -/
partial def replayConstants (names : NameSet) : M Unit := do
  for n in names do replayConstant n

end

/--
Check that all postponed constructors are identical to those generated
when we replayed the inductives.
-/
def checkPostponedConstructors : M Unit := do
  for ctor in (← get).postponedConstructors do
    match (← get).env.find? ctor, (← read).newConstants[ctor]? with
    | some (.ctorInfo info), some (.ctorInfo info') =>
      if ! (info == info') then throw <| IO.userError s!"Invalid constructor {ctor}"
    | _, _ => throw <| IO.userError s!"No such constructor {ctor}"

/--
Check that all postponed recursors are identical to those generated
when we replayed the inductives.
-/
def checkPostponedRecursors : M Unit := do
  for ctor in (← get).postponedRecursors do
    match (← get).env.find? ctor, (← read).newConstants[ctor]? with
    | some (.recInfo info), some (.recInfo info') =>
      if ! (info == info') then throw <| IO.userError s!"Invalid recursor {ctor}"
    | _, _ => throw <| IO.userError s!"No such recursor {ctor}"

end Replay'

open Replay'

/--
"Replay" some constants into an `Environment`, sending them to the kernel for checking.

Throws a `IO.userError` if the kernel rejects a constant,
or if there are malformed recursors or constructors for inductive types.
-/
def replay' (newConstants : Std.HashMap Name ConstantInfo) (env : Environment) : IO Environment := do
  let mut remaining : NameSet := ∅
  for (n, ci) in newConstants.toList do
    -- We skip unsafe constants, and also partial constants.
    -- Later we may want to handle partial constants.
    if !ci.isUnsafe && !ci.isPartial then
      remaining := remaining.insert n
  let (_, s) ← StateRefT'.run (s := { env := env.toKernelEnv, remaining }) do
    ReaderT.run (r := { newConstants }) do
      for n in remaining do
        replayConstant n
      checkPostponedConstructors
      checkPostponedRecursors
  return .ofKernelEnv s.env

end Lean.Environment

namespace ComparatorPort

/-! ### Comparator/Util.lean (verbatim) -/
def runForUsedConsts [Monad m] (info : Lean.ConstantInfo) (f : Lean.Name → m Unit) : m Unit := do
  info.type.getUsedConstants.forM f
  f info.name
  if let some val := info.value? (allowOpaque := true) then
    val.getUsedConstants.forM f
  match info with
  | .axiomInfo .. | .quotInfo .. | .defnInfo .. | .thmInfo .. | .opaqueInfo .. => return ()
  | .inductInfo info =>
    info.ctors.forM f
    info.all.forM f
  | .ctorInfo info =>
    f info.induct
  | .recInfo info =>
    info.rules.forM fun rule => do
      f rule.ctor
      rule.rhs.getUsedConstants.forM f

/-- Stand-in for lean4export's `ExportedEnv` (only `constMap` is used by comparator). -/
structure ExportedEnv where
  constMap : Std.HashMap Name ConstantInfo

/-! ### Comparator/Axioms.lean (verbatim) -/
namespace Axioms
structure Context where
  solution : ExportedEnv
  legalAxioms : Std.HashSet Lean.Name
structure State where
  worklist : Array Lean.Name
  checked : Std.HashSet Lean.Name
abbrev AxiomsM := ReaderT Context <| StateT State <| Except String

partial def loop : AxiomsM Unit := do
  if (← get).worklist.isEmpty then
    return ()
  let target ← modifyGet fun s => (s.worklist.back!, { s with worklist := s.worklist.pop })
  if (← get).checked.contains target then
    loop
  else
    let some info := (← read).solution.constMap[target]?
      | throw s!"Constant not found in solution '{target}'"
    runForUsedConsts info validateConst
    modify fun s => { s with checked := s.checked.insert target }
    loop
where
  validateConst (n : Lean.Name) : AxiomsM Unit := do
    let some info := (← read).solution.constMap[n]?
      | throw s!"Constant not found in solution '{n}'"
    if let .axiomInfo info := info then
      if !(← read).legalAxioms.contains info.name then
        throw s!"Illegal axiom detected: '{n}'"
    if !(← get).checked.contains n then
      modify fun s => { s with worklist := s.worklist.push n }
end Axioms

def checkAxioms (solution : ExportedEnv) (theoremTargets : Array Lean.Name)
    (definitionTargets : Array Lean.Name) (legalAxioms : Array Lean.Name) : Except String Unit := do
  let mut worklist := #[]
  for target in theoremTargets do
    let some solutionConst := solution.constMap[target]?
      | throw s!"Const not found in solution: '{target}'"
    let .thmInfo solutionConst := solutionConst
      | throw s!"Solution constant is not a theorem: '{target}'"
    worklist := worklist.push solutionConst.name
  for target in definitionTargets do
    let some solutionConst := solution.constMap[target]?
      | throw s!"Const not found in solution: '{target}'"
    let .defnInfo solutionConst := solutionConst
      | throw s!"Solution constant is not a definition: '{target}'"
    worklist := worklist.push solutionConst.name
  let legalAxioms := Std.HashSet.ofArray legalAxioms
  Axioms.loop.run { solution, legalAxioms } |>.run' { worklist, checked := {} }

/-! ### Comparator/Compare.lean (verbatim) -/
namespace Compare
structure Context where
  challenge : ExportedEnv
  solution : ExportedEnv
  definitionTargets : Std.HashSet Lean.Name
  theoremTargets : Std.HashSet Lean.Name
structure State where
  worklist : Array Lean.Name
  checked : Std.HashSet Lean.Name
abbrev CompareM := ReaderT Context <| StateT State <| Except String

deriving instance BEq for Lean.QuotKind
deriving instance BEq for Lean.QuotVal
deriving instance BEq for Lean.InductiveVal
deriving instance BEq for Lean.ConstantInfo

def addWorklist (n : Lean.Name) : CompareM Unit := do
  if !(← get).checked.contains n then
    modify fun s => { s with worklist := s.worklist.push n }

def addRelevantConsts (info : Lean.ConstantInfo) : CompareM Unit := do
  runForUsedConsts info addWorklist

partial def loop : CompareM Unit := do
  if (← get).worklist.isEmpty then
    return ()
  let target ← modifyGet fun s => (s.worklist.back!, { s with worklist := s.worklist.pop })
  if (← get).checked.contains target then
    loop
  else
    let some challengeConst := (← read).challenge.constMap[target]?
      | throw s!"Const not found in challenge '{target}'"
    let some solutionConst := (← read).solution.constMap[target]?
      | throw s!"Const not found in solution '{target}'"
    if (← read).definitionTargets.contains solutionConst.name
        || (← read).theoremTargets.contains solutionConst.name then
      solutionConst.type.getUsedConstants.forM addWorklist
    else
      if challengeConst != solutionConst then
        throw s!"Const does not match between challenge and target '{target}'"
      addRelevantConsts solutionConst
    modify fun s => { s with checked := s.checked.insert target }
    loop
end Compare

def definitionHoleMatches (challengeHole solutionHole : Lean.DefinitionVal) : Bool :=
  challengeHole.toConstantVal == solutionHole.toConstantVal
    && challengeHole.safety == solutionHole.safety

def compareAt (challenge solution : ExportedEnv) (theoremTargets : Array Lean.Name)
    (definitionTargets : Array Lean.Name) (primitive : Array Lean.Name) : Except String Unit := do
  let mut worklist := primitive
  for target in theoremTargets do
    let some challengeConst := challenge.constMap[target]?
      | throw s!"Const not found in challenge: '{target}'"
    let some solutionConst := solution.constMap[target]?
      | throw s!"Const not found in solution: '{target}'"
    let (challengeConst, solutionConst) ←
      match challengeConst, solutionConst with
      | .thmInfo cc, .thmInfo sc
      | .axiomInfo cc, .axiomInfo sc => pure (cc.toConstantVal, sc.toConstantVal)
      | _, _ => throw s!"Challenge and solution constant kind don't match: '{target}'"
    if challengeConst != solutionConst then
      throw s!"Challenge and solution theorem statement do not match: '{target}'"
    worklist := worklist ++ challengeConst.type.getUsedConstants
  for target in definitionTargets do
    let some challengeConst := challenge.constMap[target]?
      | throw s!"Const not found in challenge: '{target}'"
    let some solutionConst := solution.constMap[target]?
      | throw s!"Const not found in solution: '{target}'"
    let .defnInfo challengeConst := challengeConst
      | throw s!"Challenge constant is not a definition: '{target}'"
    let .defnInfo solutionConst := solutionConst
      | throw s!"Solution constant is not a definition: '{target}'"
    if !definitionHoleMatches challengeConst solutionConst then
      throw s!"Const does not match between challenge and target '{target}'"
    worklist := worklist.push solutionConst.name
  let definitionTargets := Std.HashSet.ofArray definitionTargets
  let theoremTargets := Std.HashSet.ofArray theoremTargets
  Compare.loop.run { challenge, solution, definitionTargets, theoremTargets } |>.run' { worklist, checked := {} }

/-! ### Main.lean: target lists (verbatim, minus the two names absent from v4.23) -/
def primitiveTargets : Array Name := #[
  `Nat.add, `Nat.sub, `Nat.mul, `Nat.pow, `Nat.gcd, `Nat.div, `Nat.mod, `Nat.beq, `Nat.ble, `Nat.land, `Nat.lor,
  `Nat.xor, `Nat.shiftLeft, `Nat.shiftRight, /- `String.ofList, -/ `Char.ofNat, `List, /- `eagerReduce, -/ `Nat,
  `String, `String.mk, `Char, `optParam, `autoParam, `semiOutParam, `outParam]

def builtinTargets (legalAxioms : Array Name) : Array Name :=
  if legalAxioms.contains ``Quot.sound then #[``Quot, ``Quot.mk, ``Quot.lift, ``Quot.ind] else #[]

/-! ### lean4export stand-in: the closure `lean4export MODULE -- TARGETS` would write, as a constMap -/
partial def stripMData (e : Expr) : Expr :=
  e.replace fun
    | .mdata _ b => some (stripMData b)
    | _ => none

def stripCI : ConstantInfo → ConstantInfo
  | .axiomInfo v => .axiomInfo { v with type := stripMData v.type }
  | .defnInfo v => .defnInfo { v with type := stripMData v.type, value := stripMData v.value }
  | .thmInfo v => .thmInfo { v with type := stripMData v.type, value := stripMData v.value }
  | .opaqueInfo v => .opaqueInfo { v with type := stripMData v.type, value := stripMData v.value }
  | .quotInfo v => .quotInfo { v with type := stripMData v.type }
  | .inductInfo v => .inductInfo { v with type := stripMData v.type }
  | .ctorInfo v => .ctorInfo { v with type := stripMData v.type }
  | .recInfo v => .recInfo { v with type := stripMData v.type,
                                    rules := v.rules.map fun r => { r with rhs := stripMData r.rhs } }

structure ExpState where
  visited : NameSet := {}
  out : Std.HashMap Name ConstantInfo := {}
  missing : Array Name := #[]

abbrev ExpM := ReaderT (Environment × NameMap NameSet) (StateT ExpState IO)

mutual
partial def dumpDeps (e : Expr) : ExpM Unit := do
  let e := stripMData e
  let mut nat := false
  let mut str := false
  for sub in (e.foldConsts #[] fun c acc => acc.push c) do
    dumpConstant sub
  -- literal dependencies (lean4export's dumpNatDeps / dumpStrDeps)
  if (e.find? fun | .lit (.natVal _) => true | _ => false).isSome then nat := true
  if (e.find? fun | .lit (.strVal _) => true | _ => false).isSome then str := true
  if nat then dumpConstant ``Nat
  if str then
    dumpConstant ``Char.ofNat
    dumpConstant ``String.mk

partial def dumpConstant (c : Name) : ExpM Unit := do
  let (env, recMap) ← read
  let some declar := env.find? c
    | modify fun s => { s with missing := s.missing.push c }; return
  if declar.isUnsafe || declar.isPartial || (← get).visited.contains c then
    return
  modify fun s => { s with visited := s.visited.insert c }
  let emit (ci : ConstantInfo) : ExpM Unit := modify fun s => { s with out := s.out.insert ci.name (stripCI ci) }
  match declar with
  | .axiomInfo v => dumpDeps v.type; emit declar
  | .defnInfo v => dumpDeps v.type; dumpDeps v.value; emit declar
  | .opaqueInfo v => dumpDeps v.type; dumpDeps v.value; emit declar
  | .thmInfo v => dumpDeps v.type; dumpDeps v.value; emit declar
  | .quotInfo _ =>
    dumpConstant ``Eq
    for q in [``Quot, ``Quot.mk, ``Quot.lift, ``Quot.ind] do
      let some qi@(.quotInfo _) := env.find? q
        | modify fun s => { s with missing := s.missing.push q }; return
      modify fun s => { s with visited := s.visited.insert q }
      emit qi
  | .inductInfo base =>
    let mut ctorVals : Array ConstantInfo := #[]
    let mut recNames : NameSet := {}
    let mut indVals : Array ConstantInfo := #[]
    for indName in base.all do
      let some ii := env.find? indName | modify fun s => { s with missing := s.missing.push indName }; return
      indVals := indVals.push ii
      for ctor in ii.inductiveVal!.ctors do
        let some ci := env.find? ctor | modify fun s => { s with missing := s.missing.push ctor }; return
        ctorVals := ctorVals.push ci
      modify fun s => { s with visited := s.visited.insert indName }
      dumpDeps ii.type
      if let some names := recMap.find? base.name then
        recNames := names.foldl (init := recNames) (·.insert ·)
    for cv in ctorVals do
      modify fun s => { s with visited := s.visited.insert cv.name }
      dumpDeps cv.type
    let mut recVals : Array ConstantInfo := #[]
    for rn in recNames do
      let some ri := env.find? rn | modify fun s => { s with missing := s.missing.push rn }; return
      recVals := recVals.push ri
    for rv in recVals do
      modify fun s => { s with visited := s.visited.insert rv.name }
      dumpDeps rv.type
    for rv in recVals do
      if let .recInfo rv := rv then
        for rule in rv.rules do
          dumpDeps rule.rhs
    for x in indVals ++ ctorVals ++ recVals do emit x
  | .ctorInfo v => dumpConstant v.induct
  | .recInfo v => for i in v.all do dumpConstant i
end

def exportClosure (env : Environment) (targets : Array Name) : IO (Std.HashMap Name ConstantInfo × Array Name) := do
  let mut recMap : NameMap NameSet := {}
  for (_, ci) in env.constants.map₁.toList do
    if let .recInfo rv := ci then
      for ind in rv.all do
        recMap := recMap.insert ind ((recMap.find? ind).getD {} |>.insert rv.name)
  let ((), st) ← ((targets.forM dumpConstant).run (env, recMap)).run {}
  return (st.out, st.missing)

/-! ### runBuiltinKernel (v4.23: core `Environment.replay`) -/
def runBuiltinKernel (constMap : Std.HashMap Name ConstantInfo) : IO (Option String) := do
  let quotTargets := [`Quot.mk, `Quot.lift, `Quot.ind]
  let kernelConstMap := quotTargets.foldl (init := constMap) (·.erase ·)
  let mut env ← mkEmptyEnvironment
  try
    -- Eq's closure first (the kernel's quotient initialisation looks Eq up); then everything else.
    let eqClosure ← (do
      let (m, _) ← exportClosure' kernelConstMap ``Eq
      pure m)
    let phase1 := kernelConstMap.filter fun n _ => eqClosure.contains n
    let phase2 := kernelConstMap.filter fun n _ => !eqClosure.contains n
    env ← env.replay' phase1
    env ← env.replay' phase2
  catch e =>
    return some s!"Lean default kernel rejects the solution: {e}"
  try
    let verifyTargets := `Quot :: quotTargets
    for quotTarget in verifyTargets do
      if let some info := constMap[quotTarget]? then
        let some info' := env.toKernelEnv.find? quotTarget |
          throw <| .userError s!"Could not find quotient constant in final kernel env: {quotTarget}"
        if info != info' then
          throw <| .userError s!"Quotient constant mismatch on: {quotTarget}"
    return none
  catch e =>
    return some s!"Quotient post-check rejects the solution: {e}"
where
  /-- names reachable from `root` inside `m` (by used constants), to order the replay -/
  exportClosure' (m : Std.HashMap Name ConstantInfo) (root : Name) : IO (NameSet × Unit) := do
    let mut seen : NameSet := {}
    let mut todo := #[root]
    while h : todo.size > 0 do
      let n := todo.back
      todo := todo.pop
      if seen.contains n then continue
      let some ci := m[n]? | continue
      seen := seen.insert n
      for c in ci.getUsedConstantsAsSet do todo := todo.push c
      if let .inductInfo iv := ci then
        for c in iv.ctors ++ iv.all do todo := todo.push c
    return (seen, ())

structure Config where
  challenge_module : String
  solution_module : String
  theorem_names : Array String
  definition_names : Option (Array String) := none
  permitted_axioms : Array String
  deriving FromJson, ToJson

def result (ok : Bool) (stage err : String) : IO UInt32 := do
  IO.println <| (Json.mkObj [("ok", ok), ("stage", stage), ("error", err)]).compress
  return 0

def run (dir : System.FilePath) (cfgPath : System.FilePath) : IO UInt32 := do
  let cfg ← IO.ofExcept <| (Json.parse (← IO.FS.readFile cfgPath)) >>= fromJson? (α := Config)
  let sysroot := (← IO.appPath).parent.bind (·.parent) |>.getD "."
  initSearchPath sysroot
  searchPathRef.modify (· ++ [dir])
  let theoremNames := cfg.theorem_names.map String.toName
  let definitionNames := (cfg.definition_names.getD #[]).map String.toName
  let legalAxioms := cfg.permitted_axioms.map String.toName
  let exportTargets := builtinTargets legalAxioms ++ theoremNames ++ legalAxioms ++ primitiveTargets ++ definitionNames
  let chEnv ← importModules #[{ module := cfg.challenge_module.toName }] {}
  let (chMap, chMissing) ← exportClosure chEnv exportTargets
  if !chMissing.isEmpty then
    return ← result false "export_challenge" s!"lean4export: constants not found: {chMissing}"
  let soEnv ← importModules #[{ module := cfg.solution_module.toName }] {}
  let (soMap, soMissing) ← exportClosure soEnv exportTargets
  if !soMissing.isEmpty then
    return ← result false "export_solution" s!"lean4export: constants not found: {soMissing}"
  let challenge : ExportedEnv := { constMap := chMap }
  let solution : ExportedEnv := { constMap := soMap }
  if let .error e := compareAt challenge solution (theoremNames ++ legalAxioms) definitionNames primitiveTargets then
    return ← result false "compare" e
  if let .error e := checkAxioms solution theoremNames definitionNames legalAxioms then
    return ← result false "axioms" e
  if let some e ← runBuiltinKernel soMap then
    return ← result false "kernel" e
  result true "ok" "Your solution is okay!"

end ComparatorPort

/-- lean4checker v4.23.0-rc2 `Main.lean`: `replayFromImports` / `replayFromFresh`. -/
def lean4checker (fresh : Bool) (dir module : String) : IO UInt32 := do
  let sysroot := (← IO.appPath).parent.bind (·.parent) |>.getD "."
  initSearchPath sysroot
  searchPathRef.modify (· ++ [System.FilePath.mk dir])
  let module := module.toName
  let res : Option String ← try
      if fresh then
        let env ← importModules #[{ module }] {}
        discard <| (← mkEmptyEnvironment).replay' (Std.HashMap.ofList env.constants.map₁.toList)
      else
        let mFile ← findOLean module
        let (mod, _region) ← readModuleData mFile
        let env ← importModules mod.imports {}
        let mut newConstants : Std.HashMap Name ConstantInfo := {}
        for name in mod.constNames, ci in mod.constants do
          newConstants := newConstants.insert name ci
        discard <| env.replay' newConstants
      pure none
    catch e => pure (some (toString e))
  IO.println <| (Json.mkObj [("ok", res.isNone), ("error", res.getD "")]).compress
  return 0

def main (args : List String) : IO UInt32 := do
  match args with
  | ["comparator", dir, cfg] => ComparatorPort.run dir cfg
  | ["lean4checker", dir, m] => lean4checker false dir m
  | ["lean4checker", "--fresh", dir, m] => lean4checker true dir m
  | _ => IO.eprintln "usage: fro_tools_v423.lean comparator DIR CONFIG.json | lean4checker [--fresh] DIR MODULE"; return 2
