class DataExtractorBase:

    def extract_tensor(self, name):
        raise NotImplementedError()

    def extract_weight(self, name):
        raise NotImplementedError()


class TFDataExtractor(DataExtractorBase):

    def __init__(self, sess, feed_dict):
        self.sess = sess
        self.feed_dict = feed_dict

        self.vars = { 
            v.name : v 
                for v in sess.graph.get_collection('variables')
        }

    def extract_tensor(self, op_name):
        return self.sess.graph.get_tensor_by_name(op_name + ':0').eval(feed_dict = self.feed_dict, session=self.sess)

    def extract_weight(self, var_name):
        return self.vars[var_name + ':0'].eval(session=self.sess)