def shift (c : Char) : Char :=
  if 97 ≤ c.toNat ∧ c.toNat ≤ 122 then Char.ofNat ((c.toNat - 97 + 3) % 26 + 97) else c

def cipher (s : String) : String := String.mk (s.toList.map shift)

theorem cipher_length_preservation {s : String} :
  s.any (fun c => 97 ≤ c.toNat ∧ c.toNat ≤ 122 ∨ c = ' ') →
  (cipher s).length = s.length := by
  intro _
  simp [cipher, String.length]

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
