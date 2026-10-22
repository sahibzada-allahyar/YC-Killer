def copy {n : Nat} (arr : Vector Int n) : Vector Int n :=
arr

theorem copy_spec {n : Nat} (arr : Vector Int n) :
  (copy arr).toList.length = n ∧
  ∀ i : Fin n, (copy arr)[i] = arr[i] :=
⟨by simp [copy], fun _ => rfl⟩
