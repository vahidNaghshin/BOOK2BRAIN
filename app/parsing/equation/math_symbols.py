"""Unicode math symbols to LaTeX. Explicit and reviewable: nothing is guessed."""

import re

_ENDS_WITH_CONTROL_WORD = re.compile(r"\\[A-Za-z]+$")

GREEK_LOWER = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "π": r"\pi", "ϖ": r"\varpi", "ρ": r"\rho", "ϱ": r"\varrho", "σ": r"\sigma",
    "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi", "ϕ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
}  # fmt: skip
GREEK_UPPER = {
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}  # fmt: skip

OPERATORS = {
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div", "·": r"\cdot", "⋅": r"\cdot",
    "∗": r"*", "−": "-", "–": "-", "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "∼": r"\sim",
    "≃": r"\simeq", "≅": r"\cong", "∝": r"\propto", "≪": r"\ll", "≫": r"\gg",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto", "⟶": r"\longrightarrow",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset",
    "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset", "∖": r"\setminus",
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg", "∧": r"\wedge", "∨": r"\vee",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "ℏ": r"\hbar", "ℓ": r"\ell",
    "′": "'", "″": "''", "°": r"^{\circ}", "∠": r"\angle", "⊥": r"\perp", "∥": r"\parallel",
    "∣": r"\mid", "⟨": r"\langle", "⟩": r"\rangle", "…": r"\ldots", "⋯": r"\cdots",
    "⋮": r"\vdots", "⋱": r"\ddots", "∴": r"\therefore", "∵": r"\because",
}  # fmt: skip
# Large operators: they take limits as sub/superscripts.
BIG_OPERATORS = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∬": r"\iint", "∭": r"\iiint",
                 "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap", "∐": r"\coprod"}  # fmt: skip
BLACKBOARD = {"ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
              "ℂ": r"\mathbb{C}", "ℙ": r"\mathbb{P}", "𝔼": r"\mathbb{E}"}  # fmt: skip
SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
                "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-", "ⁿ": "n"}  # fmt: skip
SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
              "₇": "7", "₈": "8", "₉": "9", "₊": "+", "₋": "-"}  # fmt: skip

# Accent characters used with <mover>: character -> LaTeX command.
ACCENTS = {"¯": r"\overline", "‾": r"\overline", "^": r"\hat", "ˆ": r"\hat", "~": r"\tilde",
           "˜": r"\tilde", "→": r"\vec", "⃗": r"\vec", "˙": r"\dot", ".": r"\dot",
           "¨": r"\ddot", "˘": r"\breve", "ˇ": r"\check"}  # fmt: skip

FUNCTION_NAMES = frozenset(
    {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh",
     "tanh", "log", "ln", "lg", "exp", "lim", "liminf", "limsup", "max", "min", "sup", "inf",
     "det", "dim", "ker", "deg", "gcd", "arg", "Pr", "hom", "mod"}
)  # fmt: skip

SYMBOLS = {**GREEK_LOWER, **GREEK_UPPER, **OPERATORS, **BLACKBOARD}


def convert_text(text: str) -> str:
    """Convert unicode math characters in a run of text to LaTeX; other characters pass through."""
    pieces: list[str] = []
    for char in text:
        if char in SYMBOLS:
            latex = SYMBOLS[char]
        elif char in BIG_OPERATORS:
            latex = BIG_OPERATORS[char]
        elif char in SUPERSCRIPTS:
            latex = "^{" + SUPERSCRIPTS[char] + "}"
        elif char in SUBSCRIPTS:
            latex = "_{" + SUBSCRIPTS[char] + "}"
        elif char == "√":
            latex = r"\sqrt"
        else:
            latex = char
        # A control word swallows the space after it, so "\alpha x" must keep its space.
        if pieces and _ENDS_WITH_CONTROL_WORD.search(pieces[-1]) and latex[:1].isalnum():
            pieces.append(" ")
        pieces.append(latex)
    return "".join(pieces)
