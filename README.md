# Playing Python with Deep Curiosity

An actor-critic algorithm learns to generate Python code that runs. It generates 100000 programs and runs them in roughly 2 minutes. It's externally rewarded 1 if the program runs and zero otherwise. Max number of steps and hence also program length is 100. It's also optionally internally rewarded for producing programs which it mispredicts the runtime behavior of.

## Number of Programs Tested Versus Number of Programs that Run (Without Curiosity)

![image](https://github.com/user-attachments/assets/6ce44d96-cb36-40d8-9632-7bfc095ebefb)

## Number of Programs Tested Versus Number of Programs that Run (With Curiosity)

![image](https://github.com/user-attachments/assets/347116b1-f312-41ca-ba62-6bfa71cf6e29)

