# Chatbot Conversation Analysis Report

*Generated March 15, 2026*

---

## 1. Dataset Overview

This report examines **1,790 conversations** containing **25,505 assistant turns** and roughly **9,583,391 words** of AI-generated text across three platforms: **Claude** (Anthropic), **Gemini** (Google), and **ChatGPT** (OpenAI).

| | Claude | Gemini | ChatGPT |
|---|---:|---:|---:|
| Conversations | 101 | 119 | 1,570 |
| Assistant turns | 1,895 | 2,228 | 21,382 |
| Total words (assistant) | 617,487 | 1,275,862 | 7,690,042 |
| Median turns per conversation | 10.0 | 24.0 | 10.0 |
| Single-exchange conversations | 14 (14%) | 19 (16%) | 281 (18%) |

ChatGPT dominates the dataset by volume (1,570 conversations), while Claude (101) and Gemini (119) provide smaller but focused windows into how the user interacts with each platform. Gemini sessions run longest per conversation (median 24 turns), suggesting more sustained multi-step interactions on that platform.

## 2. How You Use Each Platform

Topic modeling (BERTopic over all assistant turns) reveals a division of labor. The user doesn't ask the same questions everywhere — each platform has carved out its own niche.

**Claude-dominant topics:**
  - **bash, file, env, snowflake** (46 turns, 96% Claude)
  - **github, real, website, technical** (20 turns, 85% Claude)

**ChatGPT-dominant topics:**
  - **gpu, laptop, oled, zenbook** (111 turns, 80% ChatGPT)
  - **protein, sauce, cals, chicken** (80 turns, 100% ChatGPT)
  - **glucose, gsd, liver, glycogen** (68 turns, 100% ChatGPT)
  - **nausea, sugar, water, sodium** (63 turns, 98% ChatGPT)
  - **azalai, scent, molecule** (55 turns, 55% ChatGPT / 46% Gemini)
  - **bladder, urine, infection, vet** (39 turns, 100% ChatGPT)
  - **land, indigenous, mexico, spanish** (35 turns, 97% ChatGPT)
  - **bugs, insects, flies, bug** (30 turns, 100% ChatGPT)
  - **purview, fabric, data, azure** (26 turns, 73% ChatGPT)
  - **ny, york, ave, square** (21 turns, 100% ChatGPT)

**Gemini-dominant topics:**
  - **dolomites, switzerland, swiss, mountain** (39 turns, 69% Gemini)

Claude is the user's **technical co-pilot** — the go-to for writing code, building infrastructure, and working through security architecture. ChatGPT serves as the **high-volume generalist**, handling everything from health research to cooking to historical deep-dives across 1,570 conversations. Gemini handles the user's **life beyond the terminal**: veterinary questions, personal finance, and lifestyle research — but with ChatGPT's larger volume absorbing some of those categories, Gemini's topics are more evenly distributed.

The user has developed an implicit mental model of each tool's strengths and routes queries accordingly.

![Topic Distribution](plots/topic_distribution.png)

## 3. How They Write

### Length & Density

Average response length varies: Gemini 582 words, ChatGPT 364, Claude 334. Gemini's responses are **1.7× longer** than Claude's. Reading levels are comparable across all three (Flesch-Kincaid: Claude 9.3, Gemini 9.9, ChatGPT 7.9), so the difference isn't complexity — it's verbosity. Claude says what it needs to say and stops. Gemini elaborates, contextualizes, and recaps. ChatGPT lands in the middle.

![Response Length Distributions](plots/response_length_distributions.png)

### Vocabulary

Vocabulary richness (MATTR-500): ChatGPT (0.694), Gemini (0.644), Claude (0.613). ChatGPT draws from the richest lexicon per unit of text — partly driven by the sheer breadth of topics across 1,570 conversations, but the MATTR metric controls for text length. ChatGPT deploys **41,677 unique word types** vs. Gemini's 19,922 and Claude's 13,159.

![Vocabulary Comparison](plots/vocabulary_comparison.png)

### Formatting Habits

All three models are heavy formatters with different preferences. ChatGPT has the highest overall formatting density (83.6 markers per 1k words), driven by aggressive use of bullet points (25.2/1k) and bold text (31.5/1k). Gemini leans on bold text (32.8/1k) and section headers (8.8/1k), treating every response as a structured document. Claude leads in code blocks (0.68/1k vs. 0.06 and 0.01), formatting for utility over emphasis.

![Formatting Habits](plots/formatting_habits.png)

## 4. How They Talk

### Hedging & Certainty

Hedging rates per 1,000 words: Claude 4.1, Gemini 4.4, ChatGPT 4.2 — surprisingly similar overall. But the *style* of hedging differs sharply.

Claude reaches for personal uncertainty — "I think" appears 0.666 times per 1,000 words, nearly **3× more** than ChatGPT's 0.231 and **10× more** than Gemini's 0.067. Claude owns its uncertainty as a subjective stance.

Gemini hedges impersonally. It leans on "likely" (2.039/1k vs. Claude's 0.384/1k and ChatGPT's 0.622/1k), framing uncertainty as a property of the world rather than of itself. ChatGPT splits the difference, using moderate amounts of both personal and impersonal hedging.

The result: Claude sounds like a colleague thinking out loud; Gemini sounds like a textbook covering its bases; ChatGPT sounds like an eager collaborator hedging just enough to stay safe.

![Hedging Comparison](plots/hedging_comparison.png)

### Engagement & Interactivity

ChatGPT asks the most questions — **2.6 per turn** vs. Gemini's 2.3 and Claude's 1.6. But Gemini asks at least one question in **93%** of its responses (ChatGPT: 82%, Claude: 63%). Gemini is the most systematically interactive; ChatGPT asks more questions total but less consistently; Claude tends to deliver an answer and wait.

### Verbosity Ratio

For every word the user types, Claude returns roughly **11 words** (median), ChatGPT returns **18**, and Gemini returns **27** — more than double Claude. Even on identical topics (controlled for via topic modeling), the gap holds: Claude produces 534 words vs. ChatGPT's 536 and Gemini's 573.

![Verbosity Ratio](plots/verbosity_ratio.png)

### Opening Moves

How each model starts a response reveals its conversational personality:

- **Claude**: "That's", "The", "I", "This", "That" — acknowledgment and validation
- **Gemini**: "This", "That", "The", "To", "Based" — declarative and analytical
- **ChatGPT**: "Yes.", "That’s", "That", "This", "Here’s" — affirmation and agreement

Claude meets you where you are. ChatGPT validates your direction. Gemini starts building the answer immediately.

![Opening Patterns](plots/opening_patterns.png)

### The Rephrase Gap

Users rephrase their questions — retry with different wording after an unsatisfying answer — at dramatically different rates: Gemini (24.4%), Claude (14.2%), ChatGPT (4.3%). If you have to re-ask, the first answer didn't land. ChatGPT's low rate likely reflects its larger conversation count diluting the signal, but the gap remains striking even accounting for volume.

![Rephrase Rate](plots/rephrasing_rate.png)

### First-Person Voice & Disclaimers

"I" usage per 1k words: ChatGPT 7.07, Claude 5.85, Gemini 4.21. ChatGPT uses "I" most frequently but in a more agreeable register ("I'd recommend", "I think that's great"). Claude's "I" is more introspective ("I'm not sure", "I think"). Gemini avoids first-person almost entirely, maintaining an encyclopedic voice.

Disclaimer density (per 1k words): Claude 0.028, Gemini 0.008, ChatGPT 0.004. Despite ChatGPT's heavy "I" usage, it rarely hedges with explicit disclaimers. Claude is the most willing to express doubt.

## 5. Style Fingerprints

The radar chart collapses key metrics into a single visual profile for each model. The differences are consistent and mutually reinforcing:

\![Style Fingerprint](plots/style_fingerprint.png)

**Claude: The concise collaborator.** Writes shorter, formats less aggressively, hedges with personal language, asks fewer questions, and produces more varied responses (self-similarity 0.153). Behaves like a trusted colleague who gives you the answer and trusts you to ask if you need more.

**Gemini: The thorough analyst.** Highest on response length, formatting density, and verbosity. More self-similar (0.210) — a consistent "house style." Structures every response as if it might be the user's only reference on the topic. Behaves like a consultant writing a deliverable.

**ChatGPT: The agreeable generalist.** Leads with affirmation, uses the most first-person language, and has the highest self-similarity (0.238) — the most consistent voice of the three. The platform with the lowest rephrase rate, suggesting answers land effectively or that the user's rapid-fire interaction style on ChatGPT means less re-asking.

Neither style is inherently better. They serve different needs, and the user appears to have internalized this.

## 6. How You Adapt

The analysis doesn't just reveal how the bots differ — it shows how the *user* shifts behavior across platforms.

| | To Claude | To Gemini | To ChatGPT |
|---|---:|---:|---:|
| Mean message length (chars) | 124 | 131 | 78 |
| Vocabulary richness (MATTR) | 0.702 | 0.681 | 0.652 |
| Questions per turn | 1.23 | 1.28 | 0.79 |
| Politeness markers per 1k | 2.44 | 2.39 | 1.72 |

The user deploys their richest vocabulary with Claude, writes the longest messages to Gemini, and fires off the shortest, most transactional messages to ChatGPT. Politeness markers (please, thanks, sorry) are nearly identical between Claude and Gemini (~2.4 per 1k words) but drop 30% on ChatGPT (1.7) — the user treats ChatGPT more like a quick-access tool and less like a conversation partner.

The user asks the fewest questions on ChatGPT (0.79 per turn vs. 1.23-1.28 on Claude/Gemini), suggesting more directive, statement-based prompting on that platform.

\![User Language Comparison](plots/user_language_comparison.png)

## 7. Methodology Notes

- **Data sources**: Claude conversations exported as JSON with typed content blocks. Gemini conversations extracted from a Google Takeout MyActivity.html file, converted from HTML to Markdown via markdownify. ChatGPT conversations exported as JSON with a tree-structured message mapping, traced from current_node to root via parent links.
- **Conversation boundaries**: Claude and ChatGPT exports include native conversation IDs. Gemini entries (individual Q&A pairs) were grouped into conversations using a 60-minute session gap heuristic.
- **Text analysis**: All lexical and pragmatic metrics computed on assistant turns only (unless noted). Vocabulary richness uses Moving-Average Type-Token Ratio (MATTR) with a 500-word window to control for text length differences. Readability scores use the textstat library.
- **Topic modeling**: BERTopic with all-MiniLM-L6-v2 embeddings. The catch-all Topic 0 is excluded from platform-specific topic analysis.
- **Self-similarity**: Pairwise cosine similarity of 500 randomly sampled response embeddings per source.
- **Sentiment**: VADER polarity scores, bucketed into positive (>0.05), negative (<−0.05), and neutral.
- **Rephrase detection**: Consecutive user messages with cosine similarity ≥ 0.6 flagged as rephrases.
- **Topic-controlled comparison**: Metrics recomputed on a balanced random sample of 500 turns per source from the shared catch-all topic, controlling for subject-matter differences.
- **Dataset imbalance**: ChatGPT's 1,570 conversations significantly outnumber Claude (101) and Gemini (119). Per-turn and per-1k-word normalizations mitigate but do not fully eliminate volume effects.
- **All analysis runs locally** — no data is sent to external APIs. Embeddings are cached to Parquet after first computation.

## 8. What's Next

This analysis captures a snapshot of three AI relationships as they exist today. Several extensions could deepen the picture:

- **Temporal trends**: Track how response length, hedging, and topic mix evolve month-over-month. Are the models converging in style? Is the user shifting workload between platforms over time?
- **Conversation-level analysis**: Move beyond turn-level metrics to model entire conversation arcs — do certain topics produce longer sessions? Where do conversations stall?
- **Prompt engineering effects**: Correlate user prompt style (length, specificity, tone) with response quality metrics to identify what prompting strategies yield the most useful answers from each model.
- **Dataset balancing**: Collect more Claude and Gemini conversations to match ChatGPT volume for more robust cross-platform comparisons.

---

*Report generated by the Chatbot Conversation Analysis pipeline. All data processed locally.*
