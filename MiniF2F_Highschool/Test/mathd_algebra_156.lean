import Mathlib

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_156
  (x y : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀t, f t = t^4)
  (h₁ : ∀t, g t = 5 * t^2 - 6)
  (h₂ : f x = g x)
  (h₃ : f y = g y)
  (h₄ : x^2 < y^2) :
  y^2 - x^2 = 1 := by sorry