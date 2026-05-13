# QC Lab — Documentation Prose Style

This file gives the rules Claude must follow when writing prose for
the QC Lab documentation under `docs/` (Sphinx `.rst` files, the
README, or external write-ups such as collaborator handouts).
It is the prose-style counterpart to `style_guide.md`, which covers
the Python source code.

Read this file before producing or revising documentation. The rules
are derived from direct feedback on documentation drafts and should be
treated as binding.

## Table of contents

- Audience
- Naming the five objects
- "Section", not "page"
- Time-step terminology
- Adjectives and trivial sentences
- Strong design statements
- The "three layers" framing
- Lists: comprehensive vs. representative
- Worked examples in docs

---

## Audience

- The documentation primarily targets users who are new to QC Lab and
  learning to use it. Users who already use QC Lab and want to learn
  about new features are the secondary audience.
- Address the audience as "users" (or, in second-person prose, "you").
  Do not use "readers", "the reader will notice...", "for the reader
  unfamiliar with...", etc.

A consequence is that even the reference-style sections of the docs
(conventions, low-level functions, numerical constants) should remain
approachable. Don't drift into encyclopedic, telegraphic prose just
because the content is reference material.

---

## Naming the five objects

The five core objects in QC Lab — Simulation, Model, Algorithm,
Constants, and Data — are referred to in body prose as "the
Simulation object", "the Model object", and so on. The same applies to
the State object and the Parameters object, which are the State and
Parameters dictionaries threaded through every task.

- Capitalize the leading noun ("Simulation", "Model", "Algorithm",
  "Constants", "Data", "State", "Parameters") and follow it with the
  word "object".
- Use the plain-text form ("the Simulation object"), not the
  code-formatted form (``the ``Simulation`` instance``) unless the
  surrounding sentence is specifically about a Python identifier.
- When referring to the *concept* of a model, algorithm, or simulation
  in general rather than to one of the five objects, lowercase is
  preferred and consistent with the existing pages. For example:
  "QC Lab ships with two algorithms" or "a new model is added by
  subclassing Model".

Examples:

> ✅ "The Simulation object holds the Model object and the Algorithm
> object."
>
> ✅ "QC Lab ships with two algorithms tailored to Model objects defined
> in an adiabatic basis."
>
> ❌ "The simulation holds the model and the algorithm." (lowercase
> where the noun denotes one of the five objects)

---

## "Section", not "page"

In body prose, refer to a unit of documentation as a "section", not a
"page".

- "See the :ref:`Drivers <driver>` section" — yes.
- "See the :ref:`Drivers <driver>` page" — no.
- "This section describes..." — yes.
- "This page describes..." — no.

The rule applies to docstrings and `.rst` body text. It also applies to
external write-ups built from the docs (collaborator handouts, slide
decks). The rule does not apply to filenames or directory paths
(``architecture.rst`` is still a file, not a "section file") and does
not apply to mentions of the Sphinx build artifact when it is the
artifact that is at issue.

---

## Time-step terminology

QC Lab is a package for dynamical simulations, so an iteration of the
integrator is referred to as a "time step" rather than a bare "step".

- "On every update time step, the dynamics core runs the update recipe."
- "The collect time step is set by ``sim.settings.dt_collect``."

There are two kinds of time step:

- The **update time step** runs the update recipe; its length is
  ``sim.settings.dt_update``.
- The **collect time step** runs the collect recipe and updates the
  Data object; its length is ``sim.settings.dt_collect``.

Sub-divisions of a time step (RK4 sub-step, intermediate kernel calls)
are still allowed to be called "sub-steps" or "steps" in context. The
rule is about the granular iteration of the integrator forward in time.

---

## Adjectives and trivial sentences

Avoid adjectives whose meaning is unclear or whose presence is not
strictly warranted. The following words should not appear in
documentation prose unless they convey concrete information:

- "intentionally", "deliberately", "explicitly", "naturally"
- "canonical", "natural", "right", "obvious", "elegant"
- "highest-leverage", "lightweight", "powerful"
- "intentionally small", "deliberately exposed"

Replace them with a concrete description of what the design or
constraint actually is.

Trim sentences that do not deliver information. The following
constructions are common offenders and should be cut:

- "This section is important." — implied by including it.
- "It is worth noting that..." — just state the thing.
- "Note that..." (as a sentence opener) — same.
- "Of course," / "Naturally," / "Clearly," — usually drop them.

A practical filter: if you can delete a sentence and the remaining
prose loses no information, delete it.

---

## Strong design statements

Avoid making strong, generic design statements about QC Lab unless the
claim is supported by the codebase as it actually exists.

Specific examples that have been called out as incorrect:

- "A new algorithm = a new ordering of existing tasks." — This is too
  strong. A new algorithm may require new bespoke tasks (for example,
  a new update task that computes a quantity the existing tasks do not
  provide).
- "A new model = a new ingredient list." — Same problem. A new Model
  object often requires at least one new ingredient as well.
- "Every observable must be computed in an update task." — There are
  cases where the computation belongs elsewhere; check the specific
  case.

Prefer hedged phrasing that describes the common case and acknowledges
exceptions: "A new algorithm can often be expressed as a new ordering
of existing tasks, possibly augmented with new tasks for any quantity
the existing tasks do not provide."

When asserting that something *must* be a certain way, link directly to
the source code (a class definition, a critical mistake in the skill,
a CI rule) that backs the assertion.

---

## The "three layers" framing

QC Lab's design is sometimes described in terms of "three layers"
(ingredients, tasks, recipes). The documentation should not use this
framing, because the term "layer" is not defined precisely enough to
be useful and it can encourage other strong design claims of the kind
discussed in the previous section.

When discussing ingredients, tasks, and recipes, describe each of them
on its own terms, and describe the relationships between them
explicitly. If "layer" is used at all, define it concretely in the
same paragraph.

---

## Lists: comprehensive vs. representative

When the documentation includes a bulleted or tabular list of objects,
constants, slot names, or settings, make it clear to the user whether
the list is comprehensive (covers every member of the category) or
representative (includes some members for illustration).

- Comprehensive: "The list below is comprehensive: every built-in
  ingredient slot recognized by QC Lab appears in it."
- Representative: "The list below is representative of the conventions
  used by the built-in code; custom Model objects may introduce
  additional constants."

If the comprehensiveness is obvious from the surrounding context (for
example, a heading that already says "Built-in Model objects"), the
clarification is not required. If it is not obvious, the clarification
is required.

---

## Worked examples in docs

Worked examples in the documentation are illustrations of how to use
QC Lab and should not be presented as reference implementations.
A short clarifying sentence is sufficient:

> "The example below is intended as a worked example, not as a
> reference implementation."

This avoids implying that an arbitrary worked example is the canonical
way to write similar physics.

---

## Quick checklist for documentation pull requests

Before opening a documentation pull request, run through this
checklist:

- [ ] No instance of "the reader" or "readers". Replace with "users" /
      "you".
- [ ] No instance of "this page" or "the X page" in body prose.
      Replace with "this section" or "the X section".
- [ ] Each mention of one of the five objects, or the State / Parameters
      objects, uses the capitalized "Foo object" form.
- [ ] "Time step" is used instead of bare "step" when the meaning is
      "one iteration of the integrator forward in time".
- [ ] No "three layers" framing.
- [ ] No strong generic design statement of the form "X is just Y"
      unless the codebase actually enforces it.
- [ ] Each list of slots / keys / settings / objects is either clearly
      comprehensive or clearly representative.
- [ ] No adjectives from the don't-use list ("intentionally",
      "canonical", "natural", etc.) unless they convey concrete
      information.
- [ ] Worked examples are introduced as worked examples, not as
      reference implementations.
