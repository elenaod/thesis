import tensorflow

class CNN():
    """
    Class to represent the convolutional neural network used in the framework.
    
    """
    def __init__(self, **kwargs):
        self.embedding_features = kwargs["embedding_features"]
        self.semantic_features = kwargs["semantic_features"]
        self.mention_pair_features = kwargs["mention_pair_features"]
        self.activation = kwargs["activation"] if "activation" in kwargs.keys() else \
                          tensorflow.tanh
        self.filters = kwargs["filters"] if "filters" in kwargs.keys() else 280
        self.optimizer = kwargs["optimizer"] if "optimizer" in kwargs.keys() else \
                         tensorflow.train.RMSPropOptimizer(learning_rate=0.1)
        self.loss = kwargs["loss"] if "loss" in kwargs.keys() else \
                    tensorflow.losses.mean_squared_error

        hidden_layer_shape = self.filters + \
                             sum([d["shape"][3] for d in self.mention_pair_features])
        self._weights = tensorflow.Variable(
            tensorflow.random_uniform(
                shape=[hidden_layer_shape],
                dtype=tensorflow.float32
            ),
            name="hidden_weights", trainable=True)
        self._biases = tensorflow.Variable(
            tensorflow.random_uniform(
                shape=[hidden_layer_shape],
                dtype=tensorflow.float32
            ),
            name="hidden_biases", trainable=True
        )

    def _nn_mention_embedding(self, emb_features, d_features, reuse_conv):
        return tensorflow.concat(
                [tensorflow.layers.max_pooling2d(
                        tensorflow.layers.conv2d(
                                tensorflow.concat(
                                    [tensorflow.layers.max_pooling2d(
                                        tensorflow.layers.conv2d(
                                            feature["tensor"],
                                            filters=self.filters,
                                            kernel_size=[feature["kernel"], 1],
                                            activation=self.activation,
                                            reuse=reuse_conv,
                                            name="conv_1_%s" % feature["name"],
                                        ),
                                        strides=[1, 1],
                                        pool_size=[1, 1]
                                    ) for feature in emb_features],
                                    axis=1
                                ),
                            filters=self.filters,
                            kernel_size=[len(self.embedding_features), 1],
                            activation=self.activation,
                            reuse=reuse_conv,
                            name="conv_2"
                        ),
                        strides=[1, 1],
                        pool_size=[1, 1]
                )] + d_features,
                axis=3
            )

    def _nn_mention_pair_embedding(self, emb_features, d_features, mention_pair_features):
        return tensorflow.concat(
                [tensorflow.layers.max_pooling2d(
                tensorflow.layers.conv2d(
                    tensorflow.concat([
                        self._nn_mention_embedding(
                            emb_features[0],
                            d_features[0],
                            reuse_conv=None
                        ),
                        self._nn_mention_embedding(
                            emb_features[1],
                            d_features[1],
                            reuse_conv=True
                        )
                    ], axis=1),
                    filters=self.filters,
                    kernel_size=[2, 1],
                    activation=self.activation,
                    name="conv_3"
                ),
                strides=[1, 1],
                pool_size=[1, 1]
            )] + mention_pair_features,
            axis=3
        )

    def _nn_hidden_layer(self, inp, activation):
        return tensorflow.sigmoid(
            activation(
                tensorflow.add(
                    tensorflow.multiply(inp, self._weights),
                    self._biases
                )
            )
        )

    def create_nn(self):
        emb_features_placeholders = []
        d_features_placeholders = []
        mention_pair_placeholders = []

        for p in [None, None]:
            emb_features_placeholders.append([
                {
                    "tensor": tensorflow.placeholder(
                        dtype=tensorflow.float32,
                        shape=feature["shape"],
                        name="emb_feature_%s" % feature["name"]),
                    "name": feature["name"],
                    "kernel": feature["kernel"],
                    "shape": feature["shape"]
                }
                for feature in self.embedding_features
            ])
            d_features_placeholders.append([
                tensorflow.placeholder(
                    dtype=tensorflow.float32,
                    shape=feature["shape"],
                    name="semantic_feature_%s" % feature["name"])
                for feature in self.semantic_features
            ])
        mention_pair_placeholders = [
            tensorflow.placeholder(
                dtype=tensorflow.float32,
                shape=feature["shape"],
                name="mention_pair_feature_%s" % feature["name"]
            )
            for feature in self.mention_pair_features
        ]
        label_placeholder = [
            tensorflow.placeholder(
                dtype=tensorflow.float32,
                shape=[1],
                name="label"
            )
        ]

        self._eval_nn = self._nn_hidden_layer(
            self._nn_mention_pair_embedding(
                emb_features_placeholders,
                d_features_placeholders,
                mention_pair_placeholders),
            tensorflow.nn.relu
        )

        res = tensorflow.reduce_max(self._eval_nn, axis=[3, 2])
        loss = self.loss(label_placeholder, res)
        self._train_nn = self.optimizer.minimize(loss)

    def train_nn(self, session, mention_pair, label):
        graph = tensorflow.get_default_graph()

        placeholders = {}
        for f in self.embedding_features:
            placeholders[graph.get_tensor_by_name("emb_feature_%s:0" % f["name"])] = \
                session.run(f["func"](mention_pair[0]))
            placeholders[graph.get_tensor_by_name("emb_feature_%s_1:0" % f["name"])] = \
                session.run(f["func"](mention_pair[1]))
        for f in self.semantic_features:
            placeholders[graph.get_tensor_by_name("semantic_feature_%s:0" % f["name"])] = \
                session.run(f["func"](mention_pair[0]))
            placeholders[graph.get_tensor_by_name("semantic_feature_%s_1:0" % f["name"])] = \
                session.run(f["func"](mention_pair[1]))
        for f in self.mention_pair_features:
            placeholders[graph.get_tensor_by_name("mention_pair_feature_%s:0" % f["name"])] = \
                session.run(f["func"](mention_pair))
        placeholders[graph.get_tensor_by_name("label:0")] = \
            session.run(tensorflow.constant([label], dtype=tensorflow.float32))

        session.run(self._train_nn, feed_dict=placeholders)