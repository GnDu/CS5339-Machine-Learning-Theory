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

# Lecure 2

## Stochastic Gradient descent (more)

- Newton method (local search)
- Because it's taylor approximation at current point
- Get stuck at local minimum (we kinda know this)
- Locally it's the best solution but not globally
- Why not global search?
  - We can use grid search or bisection
  - Bayesian optimization using guaissian process
  - But does not scale to high dimensional problem where $\theta \in D$
    - Basically when dimension is huge
    - For example grid search, using 17 points. Scale to $17^d$ (exponentially increase!). Fundamental problem
- Global seach vs local search.
  - Local search: good for scaling, bad because local min (use this for train)
  - global search: good for global min, bad for scaling  (use this for hyperparameter tuning)
- How to avoid local min issue:
  - a family of problem that does not have this issue - convex optimisation
  - If you formulate problem as a convex optimisation, then we can scale well and avoid local min (because there is only one min)
  - For non-convx:
    - over-parameterization
    - skip connection
    - focus on this!

## Convex Optimisation

- Convex sets: Let $D \in \mathbb{R}^d$.
  - $D$ is a convex set if $\forall a,b\in D, \lambda a + (1-\lambda)b \in D, \forall \lambda \in [0,1]$
  - For example two points in a circle are in convex set because it's line ($ \lambda a + (1-\lambda)b$) will always be in the the set.
    - Non-convex set: Doughnuts, regular polygon with notches
  - $\lambda a + (1-\lambda)b \in D, \forall \lambda \in [0,1]$ is known as convex combination of $a$ and $b$. 
    - Note, not the same as linear combination ($\lambda _1 a + \lambda_{2} b $)
      - because linear combination covers all the space. Convex combination is a subset of it.
- Convex Function:
  - $f: D\rightarrow \mathbb{R}$ is a convex function for all $a$ and $b$:
    - $f(\lambda a + (1-\lambda)b) \lt \lambda f(a) + (1-\lambda)f(b)$ where $D$ is a convex set 
    - if $D$ is not convex, then LHS is undefined.
    - $f(\lambda a + (1-\lambda)b)$ - defines the function along the line segment between $a$ and $b$.
    - $\lambda f(a) + (1-\lambda)f(b)$ - defines the line segment between $f(a)$ and $f(b)$
    - If $D$ is convex set, then $\lambda f(a) + (1-\lambda)f(b)$ is bounded by $f(\lambda a + (1-\lambda)b)$
    - How do you define this beyond $d>2$?
- Epigraph(f) = $\{ (\theta, \beta) \| f(\theta)\lt\beta\}$
- $f$ is convex iff epigraph(f) is convex (as a set)
  - This allows us to define whether a function is a convex function by using sets. 
  - Every local minimum of $f$ is a global minimum for $f$ if it exists
- Proof that there can only be one global minimum in a convex function $D$
  - Defintition: $\theta$ is local min of $f$ iff $\exists \epsilon >0$ s.t:
    - $f(\theta)\leq f(\hat{\theta})$
    - $\forall\hat{\theta}\in D \cap\beta_{\epsilon}(\theta)$ where 
    - $\beta_{\epsilon}(\theta)=\{\hat{\theta}\vert\ \lvert\lvert \theta - \hat{\theta} \rvert\rvert \leq \epsilon \}$
    - $\beta_{\epsilon}(\theta)$ is basically an interval around $\theta$
      - $\{\hat{\theta}\vert \rvert \theta - \hat{\theta} \lvert \leq \epsilon \}\rightarrow$ this means the range around $\epsilon$
  - Defintition: $\theta$ is a global minimum of $f:D\in\mathbb{R}^d\rightarrow \mathbb{R}$ iff $f(\theta)\leq f(\hat{\theta}),\ \forall \hat{\theta} \in D$
    - Note, the lack of $\beta_{\epsilon}(\theta)$. Basically, we are expanding the definition to the whole of $D$ instead of just the interval.
  - Proof:
    - Let $\theta$ be a local min $f$
    - That means $\exists \epsilon >0$ s.t $\forall \hat{\theta}\in D$
    - $f(\theta)\leq f(\theta + \epsilon(\hat{\theta} - \theta))$ (upper bounded by), this is from the defintion of local minima & convexity of $D$
      - $\epsilon(\hat{\theta} - \theta)$ is basically $\beta_{\epsilon}(\theta)$
    - We simplify this: $f(\theta)\leq f(\epsilon\hat{\theta} + (1-\epsilon) \theta)$
    - assume $f$ is convex: $f(\theta) \leq \epsilon f(\hat{\theta}) + (1-\epsilon)f(\theta)$ 
      - Based on previous definition
    - Rewrite the equation: $f(\theta) - (1-\epsilon)f(\theta) \leq \epsilon f(\hat{\theta})$
    - Reduce the equation: $\epsilon f(\theta) \leq \epsilon f(\hat{\theta})$
    - Remove $\epsilon$, $f(\theta) \leq f(\hat{\theta})$
      - This covers all domain of $D$. And thus, this proves that a convex function has a global minimum
- More examples of convex function
  - $f(z) = \vert\vert z\vert\vert^2$
  - $e^2$, $e^{-z}$
  - $log \sum^d_{n=1}e^{z_n}$, $-logz$
  - $-\sqrt{z}$
  - $z^\top{a+b}$ (affine transformation)
- if $f_1...f_L$ are convex functions on D, then so are:
  - $f(\theta) = \sum \lambda_{n} f_n(\theta)$ iff $\lambda_{1}...\lambda_{n} \geq 0$
  - $f(\theta) = max f_n(\theta)$
- if $g$ is a convex function and $h$ is affine, then $f(\theta) = g(h(\theta))$ is also convex. 
  - affine = Linear + offset. $h(\theta) = a^\top\theta+b$
  - Basiically Least square regression or loss is convex
    - $f(\theta) = \frac{1}{n}\sum^n_{t=1}(\theta^\top x_n - y_n)^2$
    - $\theta^\top x_n - y_n$ is convex
    - $(.)^2$ is also affine
    - sum of convex function $\sum^n_{t=1}.$ is also affine
    - thus, least square is affine as long as $1/n$ is positive
- equivalent defs
  - $0^{th}$
