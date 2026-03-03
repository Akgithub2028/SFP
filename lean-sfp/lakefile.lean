import Lake
open Lake DSL

package «sfp» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.17.0"

@[default_target]
lean_lib «SFP» where
  srcDir := "."
