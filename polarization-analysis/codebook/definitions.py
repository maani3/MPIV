# The codebook is defined as a string that will be injected into the LLM prompt.
# It clearly lists the categories and their definitions.

CODEBOOK_DEFINITIONS = """
Here is the codebook for analyzing political polarization. You must categorize each dialogue into ONE of the following categories:

1. Strawman
   - Definition: Misrepresenting or deflecting an argument to make it easier to attack.
   - Example: Bringing up past unrelated incidents to deflect from the current topic.

2. Offensive Language
   - Definition: Words aimed to ridicule, including profanities, slurs, or foul language.

3. Absolutism
   - Definition: Treating political issues, ideas, or beliefs as entirely right or entirely wrong, presenting a false dichotomy devoid of nuance.
   - Example: "You are either with us or against us."

4. Vilification / Defamation
   - Definition: A directed personal attack on someone's character, reputation, or integrity, often using profanities or slurs.
   - Example: Calling an opponent a "thief" or "traitor".

5. Threats
   - Definition: Expressing a motive or intent to harm someone or something physically, mentally, emotionally, or socially.
   - Example: "You will pay for this."

6. Political Fear Mongering
   - Definition: Perpetuating exaggerated narratives of impending danger against an issue, party, or person to exploit fear.

7. Moral Superiority and Glorification
   - Definition: Portraying oneself or one's own group as inherently righteous and morally superior to others to gain credibility.

8. Fact-Denial / Invalidation
   - Definition: Rejecting facts, evidence, or valid opinions to undermine an opponent's credibility or justify one's own stance.

9. Otherisation
   - Definition: Isolating views or needs to differentiate and create an "us vs. them" narrative between groups or communities.

10. Neutral / Not Applicable
    - Definition: The dialogue does not fit into any of the above categories. It may be a simple statement, question, or factual reporting.
"""