### Version 0.1 - Kalman
I tried to do Kalman filtering i get nonsense results, I think I have some debbuging to do,
also maybe using directly the spike train as inputs is not a great idea. Might be smart to 
count it up in bins (20-50ms bins) and use it that way.

### Version 0

Put the linear regression code into the positionEstimator(Training).m functions. 
Expaned the algorithm to train across trials and accross time. 

The error is pretty big and we are getting predictions that hangout in the middle.

I am not sure if it is because of the way we are doing training or that it simply because we 
are not using our time series data at all when predecting and only looking at the activity of 
the last timestep.

## PreBeta

**commit 9addb728331b45f2c90f999a0f65fb3215e392dd**

Moved testingData.m code into analyseData.m and cleaned up

**commit 2237522525e8e002ad30316cdc810ac8719f3b7d**

Played around with a linear regression model on the first sample