# The codebook is defined as a string that will be injected into the LLM prompt.
# It clearly lists the categories and their definitions.

CODEBOOK_DEFINITIONS = """
### POLARIZATION ANALYSIS CODEBOOK

You must analyze the transcript using a **Multi-Label Framework**. 
Each instance of polarization consists of a pairing: **[Polarization Type] × [Polarization Code]**.

---

#### PART 1: POLARIZATION TYPES (WHO is the target?)
*Apply the "Primary Target Rule": Focus on who receives the strongest evaluative, moral, or causal language.*

1. **Personality Polarization**
   - **Target:** Individual political actors (e.g., Imran Khan, Nawaz Sharif, Judges, Journalists).
   - **Definition:** Framing individuals in extreme, moralized, or adversarial terms.
   - **Do NOT Code:** If the target is a party/institution or a policy.

2. **Party Polarization**
   - **Target:** Political parties (PTI, PMLN, PPP) or Collective Groups (The Army/Establishment, The Judiciary, The Media, The Elite).
   - **Definition:** Framing groups as adversarial, corrupt, dangerous, or illegitimate.
   - **Note:** "The Establishment" or "The Army" counts as Party Polarization (Collective Entity).

3. **Issue Polarization**
   - **Target:** Events (May 9th), Policies, Laws, or Protests.
   - **Definition:** Framing issues as existential threats, conspiracies, or moral absolutes.
   - **Do NOT Code:** If the discussion is neutral/technical.

---

#### PART 2: POLARIZATION CODES (HOW is it expressed?)

1. **Strawman**
   - **Definition:** Misrepresenting an opposing argument OR deflecting a direct question to an unrelated topic to avoid answering.
   - **Look for:** Shifting to past actions ("What about you?"), incorrect restatement of position.

2. **Absolutism**
   - **Definition:** All-or-nothing framing; presenting actors/issues as entirely right or wrong.
   - **Look for:** "Only", "Never", "Sole cause", moral finality without nuance.

3. **Vilification / Defamation**
   - **Definition:** Personal or collective attacks that ridicule, degrade, or criminalize.
   - **Look for:** Name-calling ("Traitor", "Thief", "Clown"), degrading language.

4. **Intimidation / Threats**
   - **Definition:** Threatening or dominance-asserting language intended to silence or coerce.
   - **Look for:** "Your game is over", warnings of severe consequences, crushing/erasing opponents.

5. **Political Fear Mongering**
   - **Definition:** Alarmist framing portraying actors or issues as existential threats.
   - **Look for:** Comparing protests to terrorism, claims of country destruction, conspiracy theories.

6. **Moral Superiority and Glorification**
   - **Definition:** Claims of exceptional moral, ethical, or religious superiority.
   - **Look for:** Religious legitimization, "We are the only righteous ones", beatification of leaders.

7. **Fact-Denial**
   - **Definition:** Rejecting or dismissing verifiable facts or evidence.
   - **Rule:** If evidence is explicitly rejected, use this instead of Invalidation.

8. **Invalidation**
   - **Definition:** Dismissing opposing viewpoints without engaging their substance (if no specific evidence is rejected).
   - **Look for:** Refusing to engage, brushing off valid constitutional arguments.

9. **Otherisation**
   - **Definition:** Constructing an "us vs. them" divide that delegitimizes groups.
   - **Look for:** Treating opponents as outsiders, enemies of the state, or non-citizens.

10. **Historical Grievances**
    - **Definition:** Selective invocation of past events to justify present hostility.
    - **Look for:** Digging up old conflicts to justify current anger.

11. **Offensive Language**
    - **Definition:** Use of profanity, slurs, or foul language that may not rise to the level of specific defamation but creates a hostile environment.

---
"""