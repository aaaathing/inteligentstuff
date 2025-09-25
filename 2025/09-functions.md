
# neocortex
* predict what will happen in next timestep

# bg gating
* choosing possible goals
* choosing possible actions
* choosing to store information in pfc

# semantic memory
* stored throughout the neocortex

# episodic memory
* spreading activations / priming
* when recalled, episodic memories reactivate other parts of neocortex
* also activates semantic memories

# recent memory
memory of what happened recently in last few seconds.  
* for: understanding words in context of previous words
* remembering what happened after not paying attention

# pfc
* different areas that maintain different information, like:
	* semantic information
	* reward prediction
* reasoning
	* evaluating semantic information, based on current goals and stuff
	* find relation between objects

# motor cortex
* does sequences of movements when bg gates
* at first, bg decides each movement in a sequence, and motor cortex learns it (when sleeping)
* prepares the movements to be done


# how it goes together
on Sep 6, 2025, i was at el carmelo and thinking about adding a layer that detect relations and a layer that recalls memories and pfc layers.

first, there is video input
* the first visual layer processes it into lines and edges.
* the next visual layers processes it into higher level things.
* and so on.

then, there is a layer that recieves the processed visual input and other inputs, and it combines them into a representation of a semantic thing.

the bg can gate
* it can tell pfc to store information from other layers.
* it can tell motor cortex to do movements.

the maintained representation in the pfc can also:
* activate representations in visual and semantic layers.
* activate representations in motor cortex to prepare movements.

the episodic memory can match information from visual and semantic layers (including activations from pfc) to some memories and activate the visual and semantic layers with the memories.

for reasoning, the pfc can find patterns in lower layers, and the bg can gate to store the pattern, and it can repeat for many times.


# multiple objects at once
low level areas process in parallel.
at higher level areas, the features and locations are mixed together.
but when paying attention to an object, the signals for the object get stronger and the other signals get weaker.

---


# When is BG selecting actions
ChatGPT said:

### ✅ **1. During action selection (conscious or unconscious):**

* When multiple possible actions are "on the table" (e.g., turning left or right, speaking or staying silent), the **BG is actively selecting** the most appropriate one.
* It integrates inputs from **motor**, **cognitive**, and **limbic** systems to make this choice.
* This is **constant** in dynamic environments or goal-directed behavior.

> 📌 Example: In a conversation, you're constantly choosing whether to speak, pause, or nod — BG is helping filter those actions in real time.

---

### ⚠️ **2. During automatic/habitual behavior:**

* Once an action becomes **well-learned** (like tying your shoes), the selection process becomes **less effortful**, more **automatic**, and **faster**.
* The BG is still involved, but now it works more like a **background facilitator**, engaging **"chunks" of behavior** or **habitual routines**.

> Think of it like a cached shortcut: The BG just gives the "go" signal for the whole routine, rather than evaluating each step.

---

### ⏸️ **3. During rest or passive states:**

* When you're not actively moving, deciding, or responding — say, just lying still — the BG is less engaged in action selection.
* However, it’s not “off.” It’s **monitoring**, maintaining baseline activity, and **ready to engage** if needed.
* It's also involved in **suppressing unwanted or spontaneous actions** (important for motor control).

> For example, the BG helps you *not* move when you're supposed to stay still — by **inhibiting** competing actions.

---

### 🤖 **4. In background control (e.g., posture, eye movements):**

* The BG continuously influences **subtle motor control**, like **postural adjustments** or **saccadic eye movements**, even when you're not making conscious decisions.
* These are *micro-selections* of action — often outside awareness.

---

### Summary: Is the BG always selecting actions?

| Situation                | Is BG Selecting Actions?                    | Notes                                      |
| ------------------------ | ------------------------------------------- | ------------------------------------------ |
| Active decision-making   | ✅ Yes                                       | Weighs and selects between alternatives    |
| Habitual behavior        | ✅ Yes (but faster, automatic)               | Executes learned sequences                 |
| At rest (no movement)    | ⚠️ Not selecting, but monitoring/inhibiting | Keeps motor system stable                  |
| Background motor control | ✅ Yes                                       | Controls fine-tuned, automatic adjustments |


# Chunking
ChatGPT said:

Several key brain areas are involved:

| Brain Region                       | Role in Chunking                                          |
| ---------------------------------- | --------------------------------------------------------- |
| **Basal Ganglia (BG)**             | Initiates chunks and defines their boundaries             |
| **Supplementary Motor Area (SMA)** | Plans and sequences actions; likely stores "motor chunks" |
| **Motor Cortex**                   | Executes the chunked sequence                             |
| **Cerebellum**                     | Fine-tunes and stabilizes chunked movements               |
| **Prefrontal Cortex (early on)**   | Involved in breaking down and organizing new sequences    |

---

### 🔄 How Chunking Happens (Step by Step)

#### 1. **Initial Learning**:

* Actions are **explicit**, effortful, and slow.
* Each step is processed individually, with help from the **prefrontal cortex**.

#### 2. **Repetition and Association**:

* With practice, the brain starts to **recognize patterns** across repeated actions.
* The **Basal Ganglia** begins to “tag” these patterns as a unit.

#### 3. **Formation of Chunks**:

* Neural activity shifts: instead of representing each step, the system activates **only the start and end of a chunk**.
* Middle steps become **"internalized"** and are no longer consciously controlled.

#### 4. **Automatic Execution**:

* Once chunked, sequences are **triggered by a cue** and run **automatically**.
* BG and SMA are active at the **boundaries** of the chunk (start/stop), not the middle.

---

### 📉 Neural Evidence: Start/Stop Signaling

* **Studies in rodents and primates** show that BG neurons fire **only at the beginning and end** of a learned sequence.
* This supports the idea that the BG is **"bracketing"** the chunk — marking its **boundaries**, but not managing the contents step-by-step.


# Maintaining the current action
ChatGPT said:

## 🧠 Overview: Does the Brain Keep Track of the Chunk?

### ✅ Yes — the brain **maintains a representation** of the currently active chunk until it completes.

It does this through a **distributed loop of activity** between:

* The **Basal Ganglia (BG)** — bracketing and gating the chunk
* The **Supplementary Motor Area (SMA)** and **Motor Cortex** — storing and executing the motor plan
* The **Cerebellum** — tracking timing and correcting errors
* Possibly, **working memory** or **recurrent activity** in cortical circuits

---

## 🔄 How the Brain Performs Chunks Continuously

### 🟢 1. **Chunk Initiation ("Go Signal")**

* The **BG disinhibits** the motor plan corresponding to the chunk.
* This is based on input from cortex, sensory cues, or habit triggers.
* Neural firing in the **BG and SMA** marks the *start* of the chunk.

> At this point, the brain is essentially saying: "Run this subroutine."

---

### 🧠 2. **Maintaining the Chunk in Active State**

Once the chunk begins, it is sustained by **ongoing neural activity**:

#### 🔁 a. **Recurrent Activation in Cortical Circuits**

* The **motor cortex** and **SMA** contain **recurrent neural loops** that keep the chunk's pattern "alive" throughout execution.
* Think of it like a **neural engine** that keeps running once started.

#### 📍 b. **State Tracking**

* The **SMA** and **cerebellum** help track **where you are within the chunk** (e.g., Step 3 of 5).
* This may involve **timing signals**, sensory feedback, or internal clocks.

#### 📬 c. **Error Correction and Monitoring**

* The **cerebellum** continually checks execution against expected timing or movement outcomes.
* If an error occurs, it can modulate the chunk on the fly.

---

### 🔚 3. **Chunk Termination**

* At the end of the chunk, **BG activity fires again** — marking the **chunk's stop boundary**.
* This "stop signal" may:

  * Allow the next chunk to begin (if part of a sequence).
  * Pause action (if end of task).
  * Trigger sensory feedback or evaluation.

> Studies show that in well-learned sequences, **BG neurons fire specifically at the start and end** of chunks — acting like **bookends**.

---

## 🤖 Why This Works: Chunk = Temporally Bound Motor Program

You can think of a chunk like a **compiled motor macro**:

* It runs **internally** once initiated.
* It’s **resilient** to minor distractions.
* It only requires **external input at the boundaries** (start/stop or error correction).


## ✅ So: After the basal ganglia select an action, **how does the motor cortex maintain that action across time?**

There are **three major mechanisms** involved:

---

### 1. 🌀 **Recurrent Neural Activity in the Motor Cortex**

* The **motor cortex has intrinsic recurrent connections** (neurons that excite each other in loops).
* Once a specific motor plan is activated (e.g., reaching or walking), this activity can be **self-sustaining** for a period.
* This allows the cortex to **maintain the motor pattern** across time, even without constant re-selection by the basal ganglia.

✅ This is especially true in **premotor** and **supplementary motor areas**, which are involved in **movement planning and sequencing**.

---

### 2. 🧠 **Working Memory and Top-Down Maintenance**

* The **prefrontal cortex (PFC)** and **supplementary motor areas (SMA)** help **maintain the goal or action plan** in working memory.
* This higher-level representation acts like a **“motor intention buffer”**, continually feeding the motor cortex with the intended action.

Think of this as your brain saying:

> “Keep doing this until I say stop.”

✅ The **dorsolateral prefrontal cortex (DLPFC)** holds the goal (e.g., "keep walking"), while the **motor cortex** executes and updates the actual movement.

---

### 3. 🔁 **Cortico-Basal Ganglia-Thalamo-Cortical Loop**

* Even after action selection, the **basal ganglia-thalamus loop continues to reinforce the ongoing action**.
* The **thalamus (VA/VL)** keeps exciting the motor cortex in line with the selected action, **sustaining motor cortex activation**.

✅ This loop helps ensure that the selected motor program stays active and that **competing actions remain suppressed**.

---

## 🧭 Important Supporting Systems

### 🧩 **Cerebellum: Real-Time Adjustment**

* While the motor cortex maintains the basic plan, the **cerebellum** fine-tunes it on the fly (via proprioceptive feedback).
* The cerebellum compares **intended** vs. **actual** movement and **updates motor cortex activity accordingly**.

---

### 🔁 **Sensorimotor Feedback Loops**

* As you move, sensory feedback from muscles, joints, and skin returns to the **somatosensory cortex**, which updates the **motor cortex**.
* This continuous loop lets the system know:

  > “You're still on track — or adjust now.”

---

## 🤖 Analogy: Think of it like a self-running program

* **Basal ganglia** = Selects the program to run.
* **Prefrontal cortex** = Stores the instruction to "keep running this program."
* **Motor cortex** = Executes the steps in the program, using internal loops and external feedback.
* **Thalamus & cerebellum** = Act as modulators and quality controllers, ensuring the program keeps running smoothly and accurately.


