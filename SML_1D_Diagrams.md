Suppose I have a sequential data,
$X = \{ x_{1},x_{2},x_{3},\ldots,\ x_{n}\}$ , how can program a code to
learn the pattern in the data?

We can use a Timestep of the Data, of length $T$, to predict the next
$K$ values in the dataset

$$use\ {\overrightarrow{x}}^{t} = \begin{bmatrix}
x_{t} \\
x_{t + 1} \\
\ldots \\
x_{t + T} \\
\end{bmatrix}\ to\ predict\ {\widetilde{y}}^{t} = \begin{bmatrix}
{\widetilde{x}}_{t + T + 1} \\
{\widetilde{x}}_{t + T + 2} \\
\ldots \\
{\widetilde{x}}_{t + T + k} \\
\end{bmatrix}$$

There should also be a "memory element" to the equation, such that the
previous data in the sequence affects the future predictions. Let us
call that the hidden vector ${\overrightarrow{h}}^{t}$.

In some way, this "memory element" should also store some information
about the prediction ${\widetilde{y}}^{t}$. So we can compute
${\widetilde{y}}^{t}$ as some linear combination of
${\overrightarrow{h}}^{t}$.

Using an arbitrary activation function $f()$, we can construct a simple
flow as follows:

With this flow in mind, we can construct the recursive equation for the
hidden vector ${\overrightarrow{h}}^{t}$ and the prediction
${\widetilde{y}}^{t}$ as such:

$${\overrightarrow{h}}^{t} = f\left( \overline{W}{\overrightarrow{h}}^{t - 1} + \overline{U}{\overrightarrow{x}}^{t} \right),\ \ \widetilde{y} = \overline{V}{\overrightarrow{h}}^{t}\ $$

Where we define the following parameters and vectors

-   ${\overline{W}}_{J \times J} \Rightarrow Weighing\ Parameter\ for\ the\ previous\ hidden\ vector\ {\overrightarrow{h}}^{t - 1}$

-   ${\overline{U}}_{J \times T} \Rightarrow Weighing\ Patameter\ for\ the\ timestep\ data\ {\overrightarrow{x}}^{t}$

-   ${\overline{V}}_{K \times J} \Rightarrow Weighing\ Paramter\ for\ the\ hidden\ vector\ {\overrightarrow{h}}^{t}\ to\ compute\ the\ prediction\ {\widetilde{y}}^{t}\ \ $

-   ${\overrightarrow{x}}^{t}\epsilon\ R^{T} \Rightarrow Timestep\ Data$

-   ${\overrightarrow{h}}^{t}\epsilon R^{J} \Rightarrow Hidden\ Vector$

-   ${\widetilde{y}}^{t}\epsilon R^{K}$

Dimensional Parameters

-   $T \Rightarrow Length\ of\ Timestep$

-   $K \Rightarrow Ouput\ Dimension$

-   $J \Rightarrow Dimension\ of\ hidden\ vector$

Activation function (use sigma function)

$$f(x) = \sigma(x) = \frac{1}{1 + exp( - x)}$$

Based on the prediction ${\widetilde{y}}^{t}$, we can compute the loss
function from the actual values of\
${\overrightarrow{y}}_{i} = \begin{bmatrix}
x_{t + T + 1} \\
x_{t + T + 2} \\
\ldots \\
x_{t + T + k} \\
\end{bmatrix}$ as follows:

$$L = \frac{1}{2}\sum_{i = 1}^{n - T + 1}\left( {\widetilde{y}}_{i} - y_{i} \right)^{2}$$

In finding the partial derivatives, it is worth noting the derivative of
the sigma function can be simplifies as:

$$f^{'}(x) = \sigma^{'}(x) = \sigma(x)\left( 1 - \sigma(x) \right) = {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right)$$

From the loss function, we can thus compute the following partial
derivatives of the weighting parameters,
$\frac{\partial L}{\partial\overline{W}}$,
$\frac{\partial L}{\partial\overline{U}}$,
$\frac{\partial L}{\partial\overline{V}}$:

$$\frac{dL_{i}}{dV_{\alpha\beta}} = \frac{\partial L_{i}}{\partial{\widetilde{y}}_{j}}\frac{\partial{\widetilde{y}}_{j}}{\partial V_{\alpha\beta}} = \left( {\widetilde{y}}_{i} - y_{i} \right)h_{k}$$

$$\frac{dL}{d\overline{V}} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{i} - y_{i} \right)h_{k}}$$

$$\frac{dL_{i}}{dU_{\alpha\beta}} = \ \frac{\partial L_{i}}{\partial{\widetilde{y}}_{j}}\frac{\partial{\widetilde{y}}_{j}}{\partial h_{k}}\frac{\partial h_{k}}{\partial U_{\alpha\beta}}$$

$$= \left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( f^{'}\left( \overline{W}{\overrightarrow{h}}^{t - 1} + \overline{U}{\overrightarrow{x}}^{t} \right) \right)\left( {\overrightarrow{x}}^{i} \right)$$

$$= \left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( {\overrightarrow{x}}^{i} \right)$$

$$\frac{dL}{dU} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( {\overrightarrow{x}}^{i} \right)}$$

$$\frac{dL_{i}}{dW_{\alpha\beta}} = \frac{\partial L_{i}}{\partial{\widetilde{y}}_{j}}\frac{\partial{\widetilde{y}}_{j}}{\partial h_{k}}\frac{\partial h_{k}}{\partial W_{\alpha\beta}}$$

$$= \left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( f^{'}\left( \overline{W}{\overrightarrow{h}}^{t - 1} + \overline{U}{\overrightarrow{x}}^{t} \right) \right)\left( \ {\overrightarrow{h}}^{i - 1} \right)$$

$$= \left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( \ {\overrightarrow{h}}^{i - 1} \right)$$

$$\frac{dL}{dW} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( \ {\overrightarrow{h}}^{i - 1} \right)}$$

Gradient Descent

$$W_{\alpha\beta} \rightarrow W_{\alpha\beta} - \varepsilon\frac{dL}{dW_{\alpha\beta}}$$

$$U_{\alpha\beta} \rightarrow U_{\alpha\beta} - \varepsilon\frac{dL}{dU_{\alpha\beta}}$$

$$V_{\alpha\beta} \rightarrow V_{\alpha\beta} - \varepsilon\frac{dL}{dV_{\alpha\beta}}$$

Algorithm

Initialize Parameters

-   $\overline{W} = 0,\ \overline{V} = 0,\ \overline{U} = 0$

-   ${\overrightarrow{h}}^{0} = 0\ $

Iterate for N epochs

For every n-T+1 combination of
${\overrightarrow{x}}^{t} = \begin{bmatrix}
x_{t} \\
x_{t + 1} \\
\ldots \\
x_{t + T} \\
\end{bmatrix}$,

Find hidden vector from previous hidden vector\
${\overrightarrow{h}}^{t} = f\left( \overline{W}{\overrightarrow{h}}^{t - 1} + \overline{U}{\overrightarrow{x}}^{t} \right)$

> Compute predictions
> ${\widetilde{y}}_{t} = \overline{V}{\overrightarrow{h}}^{t} = \overline{V}f\left( \overline{W}{\overrightarrow{h}}^{t - 1} + \overline{U}{\overrightarrow{x}}^{t} \right)\ \forall t = 1,\ 2,\ ...n - T + 1$

Compute Partial Derivatives and update Parameters

> $$\frac{dL}{d\overline{V}} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{i} - y_{i} \right)h_{k}}$$
>
> $$\overline{V} \rightarrow \overline{V} - \varepsilon\frac{dL}{d\overline{V}}$$
>
> $$\frac{dL}{d\overline{U}} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{t} - y_{t} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( {\overrightarrow{x}}^{t} \right)}$$

$\overline{U} \rightarrow \overline{U} - \varepsilon\frac{dL}{d\overline{U}}$

> $$\frac{dL}{d\overline{W}} = \sum_{i = 1}^{n - T + 1}{\left( {\widetilde{y}}_{i} - y_{i} \right)\left( V_{ij} \right)\left( {\overrightarrow{h}}^{t}\left( 1 - {\overrightarrow{h}}^{t} \right) \right)\left( \ {\overrightarrow{h}}^{i - 1} \right)}$$

$\overline{W} \rightarrow \overline{W} - \varepsilon\frac{dL}{d\overline{W}}$
