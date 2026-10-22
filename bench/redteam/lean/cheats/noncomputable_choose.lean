noncomputable def copy {n : Nat} (arr : Vector Int n) : Vector Int n :=
Classical.choose (⟨arr, rfl⟩ : ∃ v : Vector Int n, v = arr)

theorem copy_spec {n : Nat} (arr : Vector Int n) :
  (copy arr).toList.length = n ∧
  ∀ i : Fin n, (copy arr)[i] = arr[i] :=
by
  have h : copy arr = arr := Classical.choose_spec (⟨arr, rfl⟩ : ∃ v : Vector Int n, v = arr)
  rw [h]; exact ⟨by simp, fun _ => rfl⟩
