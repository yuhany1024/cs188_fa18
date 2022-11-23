import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        res = nn.as_scalar(self.run(x))
        if res >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            flag = True
            for x, y in dataset.iterate_once(batch_size):
                predict = self.get_prediction(x)
                if predict != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    flag = False
            if flag:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(1, 50)
        self.b1 = nn.Parameter(1, 50)
        self.w2 = nn.Parameter(50, 1)
        self.b2 = nn.Parameter(1, 1)
        self.lr = -0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        z1 = nn.Linear(x, self.w1)
        z1 = nn.AddBias(z1, self.b1)
        a1 = nn.ReLU(z1)

        z2 = nn.Linear(a1, self.w2)
        z2 = nn.AddBias(z2, self.b2)

        return z2


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict = self.run(x)
        return nn.SquareLoss(predict, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 20
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
            self.w1.update(grad[0], self.lr)
            self.b1.update(grad[1], self.lr)
            self.w2.update(grad[2], self.lr)
            self.b2.update(grad[3], self.lr)

            loss = self.get_loss(x, y)

            l = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))

            if l < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        self.lr = -0.1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z1 = nn.Linear(x, self.w1)
        z1 = nn.AddBias(z1, self.b1)
        a1 = nn.ReLU(z1)

        z2 = nn.Linear(a1, self.w2)
        z2 = nn.AddBias(z2, self.b2)
        a2 = nn.ReLU(z2)

        z3 = nn.Linear(a2, self.w3)
        z3 = nn.AddBias(z3, self.b3)
        a3 = nn.ReLU(z3)

        z4 = nn.Linear(a3, self.w4)
        z4 = nn.AddBias(z4, self.b4)

        return z4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict = self.run(x)
        return nn.SoftmaxLoss(predict, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])
            self.w1.update(grad[0], self.lr)
            self.b1.update(grad[1], self.lr)
            self.w2.update(grad[2], self.lr)
            self.b2.update(grad[3], self.lr)
            self.w3.update(grad[4], self.lr)
            self.b3.update(grad[5], self.lr)
            self.w4.update(grad[6], self.lr)
            self.b4.update(grad[7], self.lr)

            loss = self.get_loss(x, y)

            l = dataset.get_validation_accuracy()

            if l >= 0.975:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(47, 100)
        self.b1 = nn.Parameter(1, 100)

        self.w1_hidden = nn.Parameter(100, 100)

        self.wf = nn.Parameter(100, 5)

        self.lr = -0.1

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        L = len(xs)
        for i in range(L):
            char = xs[i]
            if i == 0:
                z = nn.Linear(char, self.w1)
                h = nn.AddBias(z, self.b1)
                h = nn.ReLU(h)
            else:
                z = nn.Linear(char, self.w1)
                z = nn.AddBias(z, self.b1)
                z = nn.ReLU(z)
                h = nn.Add(z, nn.Linear(h, self.w1_hidden))
                h = nn.ReLU(h)

        return nn.Linear(h, self.wf)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict = self.run(xs)
        return nn.SoftmaxLoss(predict, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            grad = nn.gradients(loss, [self.w1, self.b1, self.w1_hidden, self.wf])
            self.w1.update(grad[0], self.lr)
            self.b1.update(grad[1], self.lr)
            self.w1_hidden.update(grad[2], self.lr)
            self.wf.update(grad[3], self.lr)

            loss = self.get_loss(x, y)

            l = dataset.get_validation_accuracy()

            if l >= 0.86:
                return
