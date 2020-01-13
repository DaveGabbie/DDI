from models import *
from helper import *
import tensorflow as tf

class DDI(Model):


    def getBatches(self, data, shuffle=True):
        if shuffle: random.shuffle(data)

        for chunk in getChunks(data, self.p.batch_size):  
            batch = ddict(list)
            num = 0
            for i, bag in enumerate(chunk):
                batch['X'].append(bag['X'])
                batch['Pos1'].append(bag['Pos1'])
                batch['Pos2'].append(bag['Pos2'])
                batch['CUI'].append(bag['CUI'])
                batch['DepEdges'].append(bag['DepEdges'])
                batch['Y'].append(bag['Y'])
                old_num = num
                num += len(bag['X'])
                batch['sent_num'].append([old_num, num, i])


            yield batch



    # Reads the data from pickle file
    def load_data(self):
        data = pickle.load(open(self.p.dataset, 'rb'))
        self.voc2id = data['voc2id']
        self.type2id = data['type2id']
        self.type_num = len(data['type2id'])
        self.max_pos = data['max_pos']  # Maximum position distance
        self.num_class = len(data['rel2id'])
        self.num_deLabel = 1

        # Get Word List
        self.wrd_list = list(self.voc2id.items())  # Get vocabulary
        self.wrd_list.sort(key=lambda x: x[1])  # Sort vocabulary based on ids
        self.wrd_list, _ = zip(*self.wrd_list)
        self.data = data
        self.logger.info('Document count [{}]: {}, [{}]: {},[{}]: {}'.format('train', len(self.data['train']),'valid',len(self.data['valid']),'test', len(self.data['test'])))

    def add_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_data')  # Tokens ids of sentences
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_labels')  # Actual relation of the bag
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos1')  # Position ids wrt entity 1
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos2') 		# Position ids wrt entity 2
        self.input_cui = tf.placeholder(tf.int32, shape=[None, None], name='input_cui')#tokens cui
        self.x_len = tf.placeholder(tf.int32, shape=[None], name='input_len')  # Number of words in sentences in a batch
        self.seq_len = tf.placeholder(tf.int32, shape=(), name='seq_len')  # Max number of tokens in sentences in a batch
        self.total_sents = tf.placeholder(tf.int32, shape=(), name='total_sents')  # Total number of sentences in a batch
        self.sent_num = tf.placeholder(tf.int32, shape=[None, 3],
                                       name='sent_num')  # Stores which sentences belong to which bag
        self.de_adj_ind = tf.placeholder(tf.int64, shape=[self.num_deLabel, None, None, 2],
                                         name='de_adj_ind')  # Dependency graph information (Storing only indices and data)
        self.de_adj_data = tf.placeholder(tf.float32, shape=[self.num_deLabel, None, None], name='de_adj_data')

        self.dropout = tf.placeholder_with_default(self.p.dropout, shape=(),
                                                   name='dropout')  # Dropout used in GCN Layer
        self.rec_dropout = tf.placeholder_with_default(self.p.rec_dropout, shape=(),
                                                       name='rec_dropout')  # Dropout used in Bi-LSTM

    # Pads the data in a batch
    def padData(self, data, seq_len):
        temp = np.zeros((len(data), seq_len), np.int32)
        mask = np.zeros((len(data), seq_len), np.float32)

        for i, ele in enumerate(data):
            temp[i, :len(ele)] = ele[:seq_len]
            mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

        return temp, mask

    # Generates the one-hot representation
    def getOneHot(self, data, num_class, isprob=False):
        temp = np.zeros((len(data), num_class), np.int32)
        for i, ele in enumerate(data):
            for rel in ele:
                if isprob:
                    temp[i, rel - 1] = 1
                else:
                    temp[i, rel] = 1
        return temp

    # Pads each batch during runtime.
    def pad_dynamic(self, X, pos1, pos2, cui):
        seq_len = 0

        x_len = np.zeros((len(X)), np.int32)

        for i, x in enumerate(X):
            seq_len = max(seq_len, len(x))
            x_len[i] = len(x)

        x_pad, _ = self.padData(X, seq_len)
        pos1_pad, _ = self.padData(pos1, seq_len)
        pos2_pad, _ = self.padData(pos2, seq_len)
        cui_pad, _ = self.padData(cui, seq_len)



        return x_pad, x_len, pos1_pad, pos2_pad, cui_pad, seq_len

    def create_feed_dict(self, batch, wLabels=True, dtype='train'):  # Where putting dropout for train?
        X, Y, pos1, pos2, cui ,sent_num = batch['X'], batch['Y'], batch['Pos1'], batch[
            'Pos2'], batch['CUI'], batch['sent_num']
        total_sents = len(batch['X'])
        total_bags = len(batch['Y'])
        x_pad, x_len, pos1_pad, pos2_pad, cui_pad, seq_len = self.pad_dynamic(
            X, pos1, pos2, cui)

        y_hot = self.getOneHot(Y, self.num_class)

        feed_dict = {}
        feed_dict[self.input_x] = np.array(x_pad)
        feed_dict[self.input_pos1] = np.array(pos1_pad)
        feed_dict[self.input_pos2] = np.array(pos2_pad)
        feed_dict[self.input_cui] = np.array(cui_pad)
        feed_dict[self.x_len] = np.array(x_len)
        feed_dict[self.seq_len] = seq_len
        feed_dict[self.total_sents] = total_sents
        feed_dict[self.sent_num] = sent_num


        if wLabels: feed_dict[self.input_y] = y_hot

        feed_dict[self.de_adj_ind], \
        feed_dict[self.de_adj_data] = self.get_adj(batch['DepEdges'], total_sents, seq_len, self.num_deLabel)

        if dtype != 'train':
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rec_dropout] = 1.0
        else:
            feed_dict[self.dropout] = self.p.dropout
            feed_dict[self.rec_dropout] = self.p.rec_dropout

        return feed_dict

    # Stores the adjacency matrix as indices and data for feeding to TensorFlow
    def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
        max_edges = 0
        for edges in edgeList:
            max_edges = max(max_edges, len(edges))

        adj_mat_ind = np.zeros((max_labels, batch_size, max_edges, 2), np.int64)
        adj_mat_data = np.zeros((max_labels, batch_size, max_edges), np.float32)

        for lbl in range(max_labels):
            for i, edges in enumerate(edgeList):
                in_ind_temp, in_data_temp = [], []
                for j, (src, dest, _, _) in enumerate(edges):
                    adj_mat_ind[lbl, i, j] = (src, dest)
                    adj_mat_data[lbl, i, j] = 1.0

        return adj_mat_ind, adj_mat_data

    # GCN Layer Implementation
    def GCNLayer(self, gcn_in,  # Input to GCN Layer
                 in_dim,  # Dimension of input to GCN Layer
                 gcn_dim,  # Hidden state dimension of GCN
                 batch_size,  # Batch size
                 max_nodes,  # Maximum number of nodes in graph
                 max_labels,  # Maximum number of edge labels
                 adj_ind,  # Adjacency matrix indices
                 adj_data,  # Adjacency matrix data (all 1's)
                 w_gating=True,  # Whether to include gating in GCN
                 num_layers=1,  # Number of GCN Layers
                 name="GCN"):
        out = []
        out.append(gcn_in)

        for layer in range(num_layers):
            gcn_in = out[
                -1]  # out contains the output of all the GCN layers, intitally contains input to first GCN Layer
            if len(out) > 1: in_dim = gcn_dim  # After first iteration the in_dim = gcn_dim

            with tf.name_scope('%s-%d' % (name, layer)):
                act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
                for lbl in range(max_labels):

                    # Defining the layer and label specific parameters
                    with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
                        w_in = tf.get_variable('w_in', [in_dim, gcn_dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               regularizer=self.regularizer)
                        w_out = tf.get_variable('w_out', [in_dim, gcn_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                regularizer=self.regularizer)
                        w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 regularizer=self.regularizer)
                        b_in = tf.get_variable('b_in', initializer=np.zeros([1, gcn_dim]).astype(np.float32),
                                               regularizer=self.regularizer)
                        b_out = tf.get_variable('b_out', initializer=np.zeros([1, gcn_dim]).astype(np.float32),
                                                regularizer=self.regularizer)

                        if w_gating:
                            w_gin = tf.get_variable('w_gin', [in_dim, 1],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    regularizer=self.regularizer)
                            w_gout = tf.get_variable('w_gout', [in_dim, 1],
                                                     initializer=tf.contrib.layers.xavier_initializer(),
                                                     regularizer=self.regularizer)
                            w_gloop = tf.get_variable('w_gloop', [in_dim, 1],
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      regularizer=self.regularizer)
                            b_gin = tf.get_variable('b_gin', initializer=np.zeros([1]).astype(np.float32),
                                                    regularizer=self.regularizer)
                            b_gout = tf.get_variable('b_gout', initializer=np.zeros([1]).astype(np.float32),
                                                     regularizer=self.regularizer)

                    # Activation from in-edges
                    with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_in = tf.tensordot(gcn_in, w_in, axes=[2, 0]) + tf.expand_dims(b_in, axis=0)

                        def map_func1(i):
                            adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                      [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                            adj_mat = tf.sparse_transpose(adj_mat)
                            return tf.sparse_tensor_dense_matmul(adj_mat, inp_in[i])

                        in_t = tf.map_fn(map_func1, tf.range(batch_size), dtype=tf.float32)

                        if self.p.dropout != 1.0: in_t = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2, 0]) + tf.expand_dims(b_gin, axis=0)

                            def map_func2(i):
                                adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                          [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                                adj_mat = tf.sparse_transpose(adj_mat)
                                return tf.sparse_tensor_dense_matmul(adj_mat, inp_gin[i])

                            in_gate = tf.map_fn(map_func2, tf.range(batch_size), dtype=tf.float32)
                            in_gsig = tf.sigmoid(in_gate)
                            in_act = in_t * in_gsig
                        else:
                            in_act = in_t

                    # Activation from out-edges
                    with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_out = tf.tensordot(gcn_in, w_out, axes=[2, 0]) + tf.expand_dims(b_out, axis=0)

                        def map_func3(i):
                            adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                      [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                            return tf.sparse_tensor_dense_matmul(adj_mat, inp_out[i])

                        out_t = tf.map_fn(map_func3, tf.range(batch_size), dtype=tf.float32)
                        if self.p.dropout != 1.0: out_t = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2, 0]) + tf.expand_dims(b_gout, axis=0)

                            def map_func4(i):
                                adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                          [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                                return tf.sparse_tensor_dense_matmul(adj_mat, inp_gout[i])

                            out_gate = tf.map_fn(map_func4, tf.range(batch_size), dtype=tf.float32)
                            out_gsig = tf.sigmoid(out_gate)
                            out_act = out_t * out_gsig
                        else:
                            out_act = out_t

                    # Activation from self-loop
                    with tf.name_scope('self_loop'):
                        inp_loop = tf.tensordot(gcn_in, w_loop, axes=[2, 0])
                        if self.p.dropout != 1.0: inp_loop = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2, 0])
                            loop_gsig = tf.sigmoid(inp_gloop)
                            loop_act = inp_loop * loop_gsig
                        else:
                            loop_act = inp_loop

                    # Aggregating activations
                    act_sum += in_act + out_act + loop_act

                gcn_out = tf.nn.relu(act_sum)
                out.append(gcn_out)

        return out

    def add_model(self):
        in_wrds, in_pos1, in_pos2, in_cui = self.input_x, self.input_pos1, self.input_pos2, self.input_cui

        with tf.variable_scope('Embeddings') as scope:
            model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            embed_init = getEmbeddings(model, self.wrd_list, self.p.embed_dim)
            _wrd_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
                                              regularizer=self.regularizer)
            wrd_pad = tf.zeros([1, self.p.embed_dim])
            wrd_embeddings = tf.concat([wrd_pad, _wrd_embeddings], axis=0)


            pos1_embeddings = tf.get_variable('pos1_embeddings', [42, self.p.pos_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            pos2_embeddings = tf.get_variable('pos2_embeddings', [42, self.p.pos_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            cui_embeddings = tf.get_variable('cui_embeddings', [4000, self.p.cui_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            



        wrd_embed = tf.nn.embedding_lookup(wrd_embeddings, in_wrds)
        pos1_embed = tf.nn.embedding_lookup(pos1_embeddings, in_pos1)
        pos2_embed = tf.nn.embedding_lookup(pos2_embeddings, in_pos2)
        #cui_embed = tf.nn.embedding_lookup(cui_embeddings, in_cui)

        
        embeds = tf.concat([wrd_embed, pos1_embed, pos2_embed], axis=2)


        with tf.variable_scope('Bi-GRU') as scope:
            fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, name='FW_GRU'),
                                                    output_keep_prob=self.rec_dropout)
            bk_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, name='BW_GRU'),
                                                    output_keep_prob=self.rec_dropout)
            val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=self.x_len,
                                                         dtype=tf.float32)
            lstm_out_0 = tf.concat((state[0], state[1]), axis=1)
            lstm_out = tf.concat((val[0], val[1]), axis=2)
            lstm_out_dim = self.p.lstm_dim * 2
        de_out = self.GCNLayer(gcn_in = lstm_out, 		in_dim 	= lstm_out_dim,  gcn_dim=self.p.de_gcn_dim,
                                batch_size = self.total_sents, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel,
                                 adj_ind 	= self.de_adj_ind, 	adj_data    = self.de_adj_data, 	w_gating   = self.p.wGate,
                                 num_layers 	= self.p.de_layers, 	name 	    = "GCN_DE")
        de_out 	   = de_out[-1]
        de_out_mean  = tf.reduce_max(de_out, axis=1)			# Context  Embedding
        final = tf.concat([de_out_mean, lstm_out_0], axis=1)		# Concatenating contextual and temporal embedding
        final_dim = lstm_out_dim + self.p.de_gcn_dim
        with tf.variable_scope('FC1') as scope:
            w_rel = tf.get_variable('w_rel', [final_dim, self.num_class],
                                    initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
            b_rel = tf.get_variable('b_rel', initializer=np.zeros([self.num_class]).astype(np.float32),
                                    regularizer=self.regularizer)
            nn_out = tf.nn.xw_plus_b(final, w_rel, b_rel)

        with tf.name_scope('Accuracy') as scope:
            prob     = tf.nn.softmax(nn_out)
            y_pred   = tf.argmax(prob, 	   axis=1)
            y_actual = tf.argmax(self.input_y, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))
        return nn_out, accuracy


    def add_loss(self, nn_out):
        with tf.name_scope('Loss_op'):
            loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
            if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection
                                                                                            (tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            if self.p.opt == 'adam' and not self.p.restore:
                optimizer = tf.train.RMSPropOptimizer(self.p.lr,self.p.rho,self.p.epsilon)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
            train_op  = optimizer.minimize(loss)
        return train_op

    def __init__(self, params):
        self.p  = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p)); pprint(vars(self.p))
        self.p.batch_size = self.p.batch_size

        if self.p.l2 == 0.0: 	self.regularizer = None
        else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

        self.load_data()
        self.add_placeholders()

        nn_out, self.accuracy = self.add_model()

        self.loss      	= self.add_loss(nn_out)
        self.logits  	= tf.nn.softmax(nn_out)
        self.train_op   = self.add_optimizer(self.loss)

        tf.summary.scalar('accmain', self.accuracy)
        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None

    # Evaluate model on valid/test data
    def predict_test(self, sess, data, wLabels=True, shuffle=False, label='Evaluating on Test'):
        losses, accuracies, results, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):

            loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, dtype='test'))
            losses.    append(loss)
            accuracies.append(accuracy)

            pred_ind      = logits.argmax(axis=1)
            logit_list   += logits.tolist()
            y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
            y_pred       += pred_ind.tolist()
            y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
            bag_cnt      += len(batch['sent_num'])

            results.append(pred_ind)

            if step % 100 == 0:
                self.logger.info('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['test']), np.mean(accuracies ) *100, np.mean(losses), self.p.name))

        self.logger.info('Test Accuracy: {}'.format(accuracy))

        return np.mean(losses), results,  np.mean(accuracies ) *100, y, y_pred, logit_list, y_actual_hot
    def predict_valid(self, sess, data, wLabels=True, shuffle=False, label='Evaluating on Valid'):
        losses, accuracies, results, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):

            loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, dtype='valid'))
            losses.    append(loss)
            accuracies.append(accuracy)

            pred_ind      = logits.argmax(axis=1)
            logit_list   += logits.tolist()
            y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
            y_pred       += pred_ind.tolist()
            y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
            bag_cnt      += len(batch['sent_num'])

            results.append(pred_ind)

            if step % 100 == 0:
                self.logger.info('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['valid']), np.mean(accuracies ) *100, np.mean(losses), self.p.name))

        self.logger.info('Valid Accuracy: {}'.format(accuracy))

        return np.mean(losses), results,  np.mean(accuracies ) *100, y, y_pred, logit_list, y_actual_hot
    # Runs one epoch of training
    def run_epoch(self, sess, data, epoch, shuffle=True):
        losses, accuracies = [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            summary_str, loss, accuracy, _ = sess.run([self.merged_summ, self.loss, self.accuracy, self.train_op], feed_dict=feed)

            losses.    append(loss)
            accuracies.append(accuracy)

            bag_cnt += len(batch['sent_num'])

            if step % 10 == 0:
                self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, len(self.data['train']), np.mean
                                                                                                    (accuracies ) *100, np.mean(losses), self.p.name, self.best_train_acc))
                self.summ_writer.add_summary(summary_str, epoch *len(self.data['train']) + bag_cnt)

        accuracy = np.mean(accuracies) * 100.0
        self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
        return np.mean(losses), accuracy



    # evaluation of DDI extraction results. 4 DDI tpyes
    def result_evaluation(self, y_test, pred_test):


        pred_matrix = np.zeros((len(pred_test) ,5) ,dtype=np.int8)
        y_matrix = np.zeros((len(pred_test) ,5) ,dtype=np.int8)
        for i in range(len(y_test)):
            pred_matrix[i][pred_test[i]] = 1
            y_matrix[i][y_test[i]] = 1

        count_matrix =np.zeros((5 ,3))
        for class_idx in range(1 ,5):

            count_matrix[class_idx][0] = np.sum \
                (np.array(pred_matrix[:, class_idx]) * np.array(y_matrix[:, class_idx])  )  # tp
            count_matrix[class_idx][1] = np.sum \
                (np.array(pred_matrix[:, class_idx]) * (1 - np.array(y_matrix[:, class_idx]))  )  # fp
            count_matrix[class_idx][2] = np.sum \
                ((1 - np.array(pred_matrix[:, class_idx])) * np.array(y_matrix[:, class_idx])  )  # fn

        sumtp = sumfp = sumfn =0

        for i in range(1 ,5):
            sumtp +=count_matrix[i][0]
            sumfp +=count_matrix[i][1]
            sumfn +=count_matrix[i][2]

        precision = recall = f1 =0

        if (sumtp + sumfp) == 0:
            precision = 0.
        else:
            precision = float(sumtp) / (sumtp + sumfp)

        if (sumtp + sumfn) == 0:
            recall = 0.
        else:
            recall = float(sumtp) / (sumtp + sumfn)

        if (precision + recall) == 0.:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision ,recall ,f1
 

    # Trains the model and finally evaluates on test
    def fit(self, sess):
        self.summ_writer = tf.summary.FileWriter('tf_board/{}'.format(self.p.name), sess.graph)
        saver     = tf.train.Saver(max_to_keep=4)
        save_dir  = 'checkpoints/{}/'.format(self.p.name); make_dir(save_dir)
        res_dir   = 'results/{}/'.format(self.p.name);     make_dir(res_dir)
        save_path = os.path.join(save_dir, 'best_model')

        # Restore previously trained model
        if self.p.restore:
            saver.restore(sess, save_path)
        self.f1, self.best_train_acc = 0.0, 0.0
      

        if not self.p.only_eval:
            for epoch in range(self.p.max_epochs):
                train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
                self.logger.info \
                    ('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}\n'.format(epoch, train_loss, train_acc))
                val_loss, val_pred, val_acc, y, y_pred, logit_list, y_hot = self.predict_valid(sess,self.data['valid'])
                val_prec, val_rec, val_f1  = self.result_evaluation(y, y_pred)
                self.logger.info('Final results: Prec:{} | Rec:{} | F1:{}'.format(test_prec, test_rec, test_f1))
                # Store the model with least train loss
                if val_f1 > self.f1:
                    self.f1 = val_f1
                    saver.save(sess=sess, save_path=save_path)
                # self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_acc))

        self.logger.info('Running on Test set')
        saver.restore(sess, save_path)
        test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot = self.predict_test(sess, self.data['test'])
        test_prec, test_rec, test_f1  = self.result_evaluation(y, y_pred)
        self.logger.info('Final results: Prec:{} | Rec:{} | F1:{}'.format(test_prec, test_rec, test_f1))

    
if __name__== "__main__":

    parser = argparse.ArgumentParser \
        (description='DDI Relation Extraction')

    parser.add_argument('-data', 	 dest="dataset", 	required=True,							help='Dataset to use')
    parser.add_argument('-gpu', 	 dest="gpu", 		default='0',							help='GPU to use')
    parser.add_argument('-nGate', 	 dest="wGate", 		action='store_false',   					help='Include edgewise-gating in GCN')
    parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=200,   	type=int, 					help='Hidden state dimension of Bi-LSTM')
    parser.add_argument('-pos_dim',  dest="pos_dim", 	default=20, 			type=int, 			help='Dimension of positional embeddings')
    parser.add_argument('-cui_dim', dest="cui_dim", default=200, type=int, help='Dimension of cui embeddings')
    parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=200,   			type=int, 			help='Hidden state dimension of GCN over dependency tree')

    parser.add_argument('-de_layer', dest="de_layers", 	default=1,   			type=int, 			help='Number of layers in GCN over dependency tree')
    parser.add_argument('-drop',	 dest="dropout", 	default=0.5,  			type=float,			help='Dropout for full connected layer')
    parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.5,  			type=float,			help='Recurrent dropout for LSTM')

    parser.add_argument('-num_units', dest="num_units", 	default=400,   			type=int, 			help='Number of self_attention')
    parser.add_argument('-num_heads', dest="num_heads", 	default=8,   			type=int, 			help='Number of head')

    parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,			help='Learning rate')
    parser.add_argument('-rho',	 dest="rho", 		default=0.95,  		type=float,			help='Learning rate')
    parser.add_argument('-epsilon',	 dest="epsilon", 		default=1e-08,  		type=float,			help='Learning rate')
    parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 			help='L2 regularization')
    parser.add_argument('-epoch', 	 dest="max_epochs", 	default=6,   			type=int, 			help='Max epochs')
    parser.add_argument('-batch', 	 dest="batch_size", 	default=6,   			type=int, 			help='Batch size')
    parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 			help='Chunk size')
    parser.add_argument('-restore',	 dest="restore", 	action='store_true', 						help='Restore from the previous best saved model')
    parser.add_argument('-only_eval' ,dest="only_eval", 	action='store_true', 						help='Only Evaluate the pretrained model (skip training)')
    parser.add_argument('-opt',	 dest="opt", 		default='adam', 						help='Optimizer to use for training')

    parser.add_argument('-name', 	 dest="name", 		default='test_ ' +str(uuid.uuid4()),				help='Name of the run')
    parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,			help='Seed for randomization')
    parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 						help='Log directory')
    parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 						help='Config directory')
    # parser.add_argument('-embed_loc',dest="embed_loc", 	default='./glove/glove.6B.50d_word2vec.txt', 			help='Log directory')
    parser.add_argument('-embed_loc' ,dest="embed_loc", 	default='./glove/vec1.txt', 			help='Log directory')
    parser.add_argument('-embed_dim' ,dest="embed_dim", 	default=200, type=int,						help='Dimension of embedding')
    args = parser.parse_args()

    #if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    if not args.restore: args.name = args.name
    # Set GPU to use
    set_gpu(args.gpu)

    # Set seed
    tf.set_random_seed(args.seed)
    #random.seed(args.seed)
    np.random.seed(args.seed)

    # Create model computational graph
    model  = DDI(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)
