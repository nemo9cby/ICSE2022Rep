This is the replication package for paper submission: Towards Training Reproducible Deep Learning Models.

This replication package contains the following parts:
- experiment results/ contains the experimental results for the six open source models
- implementation/ contains the code for training the six open source models
- record-and-replay/ contains the binary format of the record-and-replay tool
- Time.xlsx contains the table for the time overhead comparison

To use the record-and-replay tool, in Linux, 
point the absolute location to LD_PRELOAD and start the training process as usual.
Check the system log: cat /var/log/syslog 

---
For the semi-formal interview:

We worked closely with ~20 practitioners, who are either senior software developers or ML scientists with Ph.D. degrees. Their tasks are to prototype DL models and/or productionalize DL models. We have conducted two separate interviews with them and each round lasted for about 2 hours. During the interview, we presented our survey on the state-of-the-art techniques towards reproducibility and gathered their feedback (reported in Section 2.3).
