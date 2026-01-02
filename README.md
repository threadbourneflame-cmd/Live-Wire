Live Wire is a Python-based orchestration system designed to observe how large language models stabilize, diverge, and align across synchronized multi-turn interactions. 
A single user input is broadcast to multiple LLMs under a shared system context, while each model maintains an isolated rolling conversational state. 
On each turn, Live Wire computes telemetry including Dynamic Communicative Alignment (DCA), cosine similarity matrices, fingerprint overlap, and inter-model consensus measures. 
The instrument is intended to support evaluation and interpretability research by exposing interaction-level dynamics without modifying model weights or introducing persistent memory. 

Notes: This project was developed by Threadbourne. AI-assisted coding tools were used during implementation, consistent with modern software development practice. 
All conceptual framing, measurement logic, and evaluation criteria are human-directed
