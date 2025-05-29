# Playing Python with Deep Curiosity

An actor-critic algorithm learns to generate Python code that runs. It generates 100000 programs and runs them in roughly 2 minutes. It's externally rewarded 1 if the program runs and zero otherwise. Max number of steps and hence also program length is 100. It's also optionally internally rewarded for producing programs which it mispredicts the runtime behavior of.

## Curiosity Reward Spike (r_c)

```
program: 66262+6662666226666662
r_c: 9.02047061920166
program: 66262+66626662266666621
r_c: 8.795820236206055
program: 66262+66626662266666621]
r_c: 0.00010251473577227443
program: 66262+66626662266666621]2
r_c: 8.40390202938579e-05
```

## Curiosity and External Rewards Over Time

![image](https://github.com/user-attachments/assets/85ab936a-b70e-473e-99dd-31bd97ce5f6a)

![image](https://github.com/user-attachments/assets/79c91245-e33b-41e2-9ea3-ccf84542c65b)

## Number of Programs Tested Versus Number of Programs that Run (Without Curiosity)

![image](https://github.com/user-attachments/assets/6ce44d96-cb36-40d8-9632-7bfc095ebefb)

## Number of Programs Tested Versus Number of Programs that Run (With Curiosity)

![image](https://github.com/user-attachments/assets/347116b1-f312-41ca-ba62-6bfa71cf6e29)

