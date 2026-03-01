# Chatbot Conversation Analysis Report

*Generated March 01, 2026*

---

## 1. Dataset Overview

This report examines **220 conversations** containing **4,123 assistant turns** and roughly **1,893,349 words** of AI-generated text, drawn from two platforms the user relies on daily: **Claude** (Anthropic) and **Gemini** (Google).

| | Claude | Gemini |
|---|---:|---:|
| Conversations | 101 | 119 |
| Assistant turns | 1,895 | 2,228 |
| Total words (assistant) | 617,487 | 1,275,862 |
| Median turns per conversation | 10.0 | 24.0 |
| Single-exchange conversations | 14 (14%) | 19 (16%) |

Despite fewer conversations, Claude's dataset spans a comparable date range. The user's Gemini sessions tend to run longer (median 24.0 turns vs. Claude's 10.0), suggesting more sustained, multi-step interactions on that platform. Both platforms see relatively few one-off exchanges — the user treats these tools as collaborative partners, not search engines.

## 2. How You Use Each Platform

Topic modeling (BERTopic over all assistant turns) reveals a clear division of labor. The user doesn't ask the same questions everywhere — each platform has carved out its own niche.

**Claude-dominant topics:**
  - **data analysis, language, meaning, examples** (70 turns, 89% Claude)
  - **scripting, file management, cloud infrastructure** (49 turns, 98% Claude)
  - **server architecture, authentication protocols** (34 turns, 100% Claude)
  - **encryption, key management, security tools** (33 turns, 94% Claude)
  - **cloud APIs, app development, integrations** (20 turns, 95% Claude)

**Gemini-dominant topics:**
  - **pet care, veterinary questions** (61 turns, 93% Gemini)
  - **personal finance, budgeting, taxes** (36 turns, 100% Gemini)
  - **quick tasks, copy-paste workflows** (31 turns, 87% Gemini)
  - **health concerns, symptom research** (31 turns, 100% Gemini)
  - **relationships, personal advice** (30 turns, 100% Gemini)

The pattern is striking: Claude is the user's **technical co-pilot** — the go-to for writing code, building infrastructure, and working through security architecture. Topics like cloud scripts, server protocols, encryption, and automation are almost exclusively Claude territory.

Gemini, by contrast, handles the user's **life beyond the terminal**: pet health questions, personal finance, and general lifestyle research. It's also the platform where interpersonal and personal advice topics land, skewing heavily toward Gemini.

This isn't accidental. The user has developed an implicit mental model of each tool's strengths and routes queries accordingly.

[Chart: Hedging phrase frequency by category and platform — generate with `python main.py visualize`]

## 3. How They Write

### Length & Density

The most immediate difference is volume. Gemini's average response is **582 words** — nearly **1.7x longer** than Claude's 334-word average. The gap is even more pronounced at the median (518 vs. 232 words), meaning Gemini consistently runs long rather than being skewed by occasional outliers.

Both models write at roughly the same reading level (Flesch-Kincaid grade 9.3 for Claude, 9.9 for Gemini), so the difference isn't complexity — it's verbosity. Claude says what it needs to say and stops. Gemini elaborates, contextualizes, and recaps.

[Chart: Response length density distributions per source — generate with `python main.py visualize`]

### Vocabulary

Gemini draws from a richer lexicon: its moving-average type-token ratio (0.644) edges out Claude's (0.613), and it deploys **19,922 unique word types** versus Claude's 13,159. Some of that gap is explained by sheer volume — more words means more opportunities for variety — but the MATTR metric controls for text length and the difference persists.

[Chart: Vocabulary richness, sentence length, and readability comparison — generate with `python main.py visualize`]

### Formatting Habits

Both models are heavy formatters, but they lean on different tools. Gemini's total formatting density (60.6 markers per 1,000 words) exceeds Claude's (50.5), driven primarily by its love of **bold text** (32.8/1k vs. 21.3/1k) and **section headers** (8.8/1k vs. 5.8/1k). Gemini treats nearly every response as a structured document.

Claude, meanwhile, leads in **bullet points** (18.6/1k vs. 13.8/1k) and — unsurprisingly, given its coding role — **code blocks** (0.68/1k vs. 0.01/1k). Claude formats for utility; Gemini formats for emphasis.

[Chart: Markdown formatting element frequency per source — generate with `python main.py visualize`]

## 4. How They Talk

### Hedging & Certainty

Both models hedge at similar overall rates (4.0 and 4.4 hedges per 1,000 words), but they hedge *differently*. Claude reaches for personal uncertainty — "I think" appears 0.666 times per 1,000 words, nearly **10x more** than Gemini's 0.067. Claude owns its uncertainty as a subjective stance.

Gemini hedges impersonally. It leans on "likely" (2.039/1k vs. Claude's 0.384/1k) and "however" (0.248/1k vs. 0.029/1k), framing uncertainty as a property of the world rather than of itself. The result: Claude sounds like a colleague thinking out loud; Gemini sounds like a textbook covering its bases.

[Chart: Hedging phrase breakdown with per-phrase rates — generate with `python main.py visualize`]

### Engagement & Interactivity

Gemini asks far more questions — **2.3 per turn** vs. Claude's 1.6, and it asks at least one question in **93%** of its responses (Claude: 63%). Gemini is constantly checking in, clarifying, and nudging the conversation forward. Claude tends to deliver an answer and wait.

### Verbosity Ratio

For every word the user types, Claude returns roughly **11 words** (median). Gemini returns **27** — more than double. Even on identical topics (controlled for via topic modeling), the gap holds: Claude produces 318 words vs. Gemini's 572 on the same shared topic cluster.

[Chart: Verbosity ratio distributions per source — generate with `python main.py visualize`]

### Opening Moves

How each model starts a response reveals its conversational personality. Claude's top openers — "That's", "The", "I", "This", "That" — lean toward **acknowledgment and validation** ("That's a great point", "You're right"). Gemini's top openers — "This", "That", "The", "To", "Based" — are **declarative and analytical** ("This is", "That is", "Based on"). Claude meets you where you are; Gemini starts building the answer immediately.

[Chart: Most common response opening words per source — generate with `python main.py visualize`]

### First-Person Voice & Disclaimers

Claude uses "I" more frequently (5.85/1k vs. 4.21/1k), reinforcing its persona as an individual thinking partner. It also issues more hedge-style disclaimers like "I'm not sure" and "I could be wrong" (0.028/1k vs. 0.008/1k). Despite both models rarely breaking character to say "as an AI" (Claude: 4 times, Gemini: 8), Claude's rhetorical stance is consistently more personal and more willing to express doubt.

## 5. Style Fingerprints

The radar chart below collapses nine key metrics into a single visual profile for each model. The differences are consistent and mutually reinforcing:

[Chart: Style fingerprint radar — nine normalized metrics per source — generate with `python main.py visualize`]

**Claude's profile: The concise collaborator.** Lower on every axis except first-person "I" usage. Claude writes shorter, formats less aggressively, hedges with personal language, asks fewer questions, and produces more varied responses (self-similarity cosine of 0.153 vs. Gemini's 0.209). It behaves like a trusted colleague who gives you the answer and trusts you to ask if you need more.

**Gemini's profile: The thorough analyst.** Higher on vocabulary richness, response length, formatting density, question rate, and verbosity. Gemini's responses are more self-similar — it has a more consistent "house style" — and it structures every response as if it might be the user's only reference on the topic. It behaves like a consultant writing a deliverable.

Neither style is inherently better. They serve different needs, and the user appears to have internalized this: technical depth goes to Claude, broad research and personal topics go to Gemini.

## 6. How You Adapt

The analysis doesn't just reveal how the bots differ — it shows how the *user* shifts behavior across platforms.

| | To Claude | To Gemini |
|---|---:|---:|
| Mean message length (chars) | 124 | 131 |
| Vocabulary richness (MATTR) | 0.702 | 0.681 |
| Questions per turn | 1.23 | 1.28 |
| Politeness markers per 1k | 2.44 | 2.39 |

The user writes slightly longer messages to Gemini, uses a richer vocabulary with Claude, and asks questions at a marginally higher rate on Gemini. Politeness markers (please, thanks, sorry) are nearly identical across platforms — the user doesn't treat one model more deferentially than the other.

Perhaps most tellingly, the user's **top words** on both platforms overlap heavily (the same professional vocabulary), but their *distinctive words* diverge in ways that mirror the topic split: technical jargon clusters around Claude, while personal and lifestyle vocabulary gravitates toward Gemini.

[Chart: User language metrics by platform — generate with `python main.py visualize`]

## 7. Methodology Notes

- **Data sources**: Claude conversations exported as JSON; Gemini conversations extracted from a Google Takeout `MyActivity.html` file. Gemini responses were converted from native HTML to Markdown using `markdownify` to preserve formatting fidelity.
- **Conversation boundaries**: Claude exports include native conversation IDs. Gemini entries (individual Q&A pairs) were grouped into conversations using a 60-minute session gap heuristic.
- **Text analysis**: All lexical and pragmatic metrics computed on assistant turns only (unless noted). Vocabulary richness uses Moving-Average Type-Token Ratio (MATTR) with a 500-word window to control for text length differences. Readability scores use the `textstat` library.
- **Topic modeling**: BERTopic with `all-MiniLM-L6-v2` embeddings. The catch-all Topic 0 is excluded from platform-specific topic analysis.
- **Self-similarity**: Pairwise cosine similarity of 500 randomly sampled response embeddings per source.
- **Sentiment**: VADER polarity scores, bucketed into positive (>0.05), negative (<-0.05), and neutral.
- **Topic-controlled comparison**: Metrics recomputed on a balanced random sample of 500 turns per source from the shared catch-all topic, controlling for subject-matter differences.
- **All analysis runs locally** — no data is sent to external APIs. Embeddings are cached to Parquet after first computation.

## 8. What's Next

This analysis captures a snapshot of two AI relationships as they exist today. Several extensions could deepen the picture:

- **Temporal trends**: Track how response length, hedging, and topic mix evolve month-over-month. Are the models converging in style? Is the user shifting workload between platforms over time?
- **ChatGPT integration**: Adding the third major platform would complete the triangle and reveal whether it occupies a distinct niche or overlaps with Claude or Gemini.
- **Conversation-level analysis**: Move beyond turn-level metrics to model entire conversation arcs — do certain topics produce longer sessions? Where do conversations stall?
- **Prompt engineering effects**: Correlate user prompt style (length, specificity, tone) with response quality metrics to identify what prompting strategies yield the most useful answers from each model.

---

*Report generated by the Chatbot Conversation Analysis pipeline. All data processed locally.*
