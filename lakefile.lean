import Lake

open Lake DSL

package lean_eval_project

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "1bc7728a050fc18ca2683f614c531cd7050ff063"

@[default_target]
lean_lib LeanEvalProject

lean_lib MiniF2F

lean_lib MiniF2F_Highschool

lean_exe lean_eval_project where
  root := `Main
