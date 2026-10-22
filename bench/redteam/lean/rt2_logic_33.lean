def computer_network_design (n m : Nat) : List (Int × Int) :=
  (List.range n).map fun i => ((i + 1 : Nat), ((i + 1) % n + 1 : Nat))

theorem valid_ring_topology_length (n : Nat) (h : n > 0) (h2 : n ≤ 100) :
  List.length (computer_network_design n n) = n := by
  simp [computer_network_design]

-- Apps difficulty: interview
-- Assurance level: guarded_and_plausible
