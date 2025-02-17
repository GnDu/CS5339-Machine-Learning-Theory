- [Lecture 1](#lecture-1)
  - [Training Loss](#training-loss)
  - [Gradient Descent](#gradient-descent)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)

# Lecture 1

- Goal of ML: minimise expected loss when deployed outside of training environment.
- Goal of training: minimize training loss.
  - Note, different from ML! 
  - Training and reality has a gap
- Generalized to: expected loss = training loss + generalisation error
  - generalisation error tends to 0 as training data increases (in terms of diversity and quantity)
  - typically by $O(\sqrt{\frac{1}{N}})$
- Therefore, we focus on minimizing training loss as expected loss approximate to training loss as training points increase

## Training Loss

- minimise training loss $f(\theta)$ where $\theta$ is the parameter of the model, $f(\theta) \in \mathbb{R}$
- typically, $f(\theta) = \frac{1}{N}\sum^{t=1}_n f_t(\theta)$
  - $f_t$ is the training loss for $t$-th data point $t$, $x_t$
  - $f_t(\theta) = l(h_{\theta}(x_t), y_t)$ where:
    - $h_{\theta}(x_t)$ is the predicted output given $x_t$ or output
      - Linear model: $\theta^Tx$ - dot product
    - $y_t$ is the target _vector_
    - $l$ is the function to measure the difference between $h_{\theta}(x_t)$ and $y_t$. Examples of $l$
      - square loss: $l(a,b) = (a-b)^2$
      - cross-entropy loss, etc

## Gradient Descent

- Gradient descent to minimise loss:
  1) initialise $\theta^0$ where $0$ refers to the step count
  2) update $\theta^{k+1} = \theta^k - \alpha \nabla f(\theta^{k})$
     - $\alpha$ - learning rate or step size
     - $\nabla f(\theta^{k})$ - gradient (derivative of $f(\theta)$) of $\theta$, $\theta \in \mathbb{R}^d$
       - Because it's a vector, the differentiation occurs element wise w.r.t to vector element: $\frac{\delta f(\theta)}{\delta \theta_1}, \frac{\delta f(\theta)}{\delta \theta_2} ...$
  - "Step down hill" concept - tends to local minima
- effects of $\alpha$:
  - too high: overshoot and swing around the local minima or worse
  - too low: converge very slow
- Derivation: 
  - why it works
  - Setup: we are at $\theta^k$, we want to move 'down hill', $\theta^k + \epsilon \Delta$ s.t $f(\theta^k + \epsilon \Delta) < f(\theta^k)$ (basically more minmised)
    - $\epsilon$ is small
    - arbitrary $\Delta \in \mathbb{R}^d$ because it must be the same shape as $\theta$ s.t $\Vert \Delta \Vert = 1$
      - Just move towards the direction. Don't care about magnitude, that's what the $\epsilon$ is for.
    - Taylor approximation (first order)
      - $f(\theta^k + \epsilon \Delta) {taylor \atop \approx} f(\theta^k) + \epsilon \nabla f(\theta^k)^T \Delta$
    - minimise by choosing $\Delta$
    - We don't know what $\Delta$ to choose for LHS so we use first order taylor approximation to decide it for us as an approximation
      - Looking at the RHS, $\Delta$ is not dependent on $f(\theta^k)$ so we can ignore it.
      - $\epsilon$ is just a scalar, can be seen as a constant.
      - Therefore:
        - $\underset{\Delta}{argmin}  \nabla f(\theta^k)^T \Delta = \underset{\Delta}{argmin} \Vert \nabla f(\theta^k) \Vert \Vert\Delta\Vert cos(\nabla f(\theta^k), \Delta)$
          - Recall your dot product": $a^Tb = \Vert a \Vert \Vert b \Vert cos(a,b)$
          - $(a,b)$ is the angle between $a$ and $b$
        - $\Vert \nabla f(\theta^k) \Vert$ does not depend on $\Delta$
        - $\Delta$ is a unit vector thus $\Vert\Delta\Vert = 1$
        - We can simplify it to: $\underset{\Delta}{argmin} \nabla f(\theta^k)^T \Delta = \underset{\Delta}{argmin}\ cos(\nabla f(\theta^k), \Delta)$
          - $cos(a,b) \in [-1,1]$, $cos(u)=-1$ is minimised when $u=\pi$
          - Basically it goes opposite of the gradient direction
        - Simplifying it again: $\underset{\Delta}{argmin} \nabla f(\theta^k)^T \Delta = \frac{-\nabla f(\theta^k)}{\Vert \nabla f(\theta^k) \Vert}$
          - We need to normalise, thus dividing by the norm
    - So let's rewind:
      - $\theta^{k+1} = \theta^k + \epsilon \Delta$
      - $\theta^{k+1} = \theta^k - \epsilon \frac{-\nabla f(\theta^k)}{\Vert \nabla f(\theta^k) \Vert}$ (because of the first order approximation we did)
      - $\theta^{k+1} = \theta^k - \alpha \nabla f(\theta^k)$ (let $\alpha = \frac{\epsilon}{\Vert \nabla f(\theta^k) \Vert}$)
      - Viola, this is how we get gradient descent!
  - So under what condition do we see the 'downhill' effect or under what condition is $f(\theta^{k+1}) < f(\theta^k)$
    - step size, $\alpha$ when sufficiently small
    - $\nabla f(\theta^k)  \neq 0$
      - Proof:
        - Let us taylor's theorem.
        - $f(\theta^k + \epsilon \Delta) = f(\theta^k) + \nabla f(\theta^k)^T \Delta + g(\Delta)\Vert \Delta \Vert$ where $\underset{\Delta \rightarrow 0}{lim}\ g(\Delta)=0$
          - as $\Delta \rightarrow 0$, basically, the gradient gets smaller and smaller.
        - Now, let us define:
          - $\Delta =  -\alpha \nabla f(\theta^k)$
          - $\phi(\alpha) = -g(\alpha \nabla f(\theta^k))$
        - We can rewrite it as : $f(\theta^k -\alpha \nabla f(\theta^k)) = f(\theta^k) - \alpha \Vert\nabla f(\theta^k)\Vert^2 +\phi(\alpha)\alpha \Vert \nabla f(\theta^k) \Vert$
        - Group: $f(\theta^k + \epsilon \Delta) =  f(\theta^k) - \alpha \Vert\nabla f(\theta^k)\Vert \ . \  [\Vert \nabla f(\theta^k) \Vert - \phi(\alpha)]$
          - if $\nabla f(\theta^k) \neq 0$ and $\alpha\gt 0$, $\alpha \Vert\nabla f(\theta^k)\Vert > 0$
          - $\alpha$ is small, $[\Vert \nabla f(\theta^k) \Vert - \phi(\alpha)] > 0$
            - There exists $\alpha'$ such that for all $\alpha < \alpha'$, $\phi(\alpha) < \Vert \nabla f(\theta^k) \Vert$ as long as $\nabla f(\theta^k)$ is non-zero
            - Because, $\phi(\alpha)=0$ as $\alpha \rightarrow 0$ due to the defintion of $g$
    - Side note: scientific process: experiement inform hypothesis, maths confirm hypothesis

## Stochastic Gradient Descent

- Gradient descent: $\nabla f(\theta) = f(\theta) = \frac{1}{N}\sum^{t=1}_n \nabla f_t(\theta)$
  - Expensive, need to compute for all $n$ where it could be practically impossible to compute.
- So let's use one data point:
  - update $\theta^{k+1} = \theta^k - \alpha \nabla f_t(\theta^{k})$
  - where $f_t$ is the loss function for $x_t$. $x_t$ can be sampled or gotten sequentially.
    - Good: less computationally expensive
    - Bad: Really long time to converge
- Compromise: Mini-batch SGD
  - $\theta^{k+1} = \theta^k - \alpha [\frac{1}{I} \sum_{i\in I}\nabla f_i(\theta^{k})]$ for index set $I$