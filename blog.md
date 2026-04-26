# Agentic Hire Arena

We aren't just teaching AI to click buttons. We are using OpenEnv to post-train agents that cannot be scammed, fooled, or bullied.

Current models are trained to be incredibly polite and helpful. That makes them great chatbots, but terrible autonomous agents for real-world systems. If a human lies to them, they believe it. If a human pressures them, they cave. They are 'yes-men.'

The next big leap in post-training isn't just teaching AI how to use tools or write code. It is about post-training AI to have a backbone. To build reliable automation, we need to teach agents how to survive human manipulation.

So we built **AgentHire Arena**. On the surface, it looks like a hiring game. Underneath, it’s a trap specifically designed to break sycophantic AIs.

In this environment, the agent is given a budget to hire a team, but the system actively tries to exploit it:
* Feeding it fake numerical resumes.
* Deploying candidates coached to lie perfectly in interviews.
* Adding a hostile AI manager (NPC) that randomly pressures the agent to rush decisions and skip verification.

But environments aren't just for testing—they are for post-training. 

We didn't just build this to watch models fail. We built it to generate the exact reward signals needed to fix them. By piping this environment into a training loop, we taught a small, open-source model to do what massive generalist models like GPT-4o couldn't. The system rewards the agent for spending its budget to dig for the truth, and heavily penalizes it for yielding to human pressure.