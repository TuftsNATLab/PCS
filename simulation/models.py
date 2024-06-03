import numpy as np
import os, sys

INF = float('inf')

# Scale total iterations of all models.
_SCALE_TOTAL_ITER = 0.1

class Model(object):
    def __init__(self, name, total_iter, times_per_iter):
        times_per_iter = [x / 1000000. for x in times_per_iter]
        # Constants
        self.name = name
        self.total_iter = int(total_iter * _SCALE_TOTAL_ITER)
        self.times_per_iter = times_per_iter
        self.throughputs = [1 / float(t) for t in times_per_iter]
        self.speedups = [times_per_iter[0] / float(t) for t in times_per_iter]

    def get_speed_up_dic(self):
        
        
        speed_up_dic = {0: 0}
        for i in range(len(self.speedups)):
            speed_up_dic[i+1] = self.speedups[i]
        return speed_up_dic


    @property
    def max_gpus(self):
        return len(self.speedups)

    def throughput(self, num_gpus):
        if num_gpus <= 0:
            return 0
        return self.throughputs[num_gpus - 1]
    def speedup(self, num_gpus):
        if num_gpus <= 0:
            return -INF
        return self.speedups[num_gpus - 1]

################################################################################

class LinearModel(Model):
    def __init__(self):
        super(LinearModel, self).__init__('LinearModel',
                153740040 // 256,
                [1.0/(i+1) for i in range(1000)])


class VggnetModel256(Model):
    def __init__(self):
        super(VggnetModel256, self).__init__('VggnetModel256',
                153740040 // 256,
                [3020532, 1659208, 1263951, 868694,
                765689, 662684, 559679, 456675])

class GooglenetModel128(Model):
    def __init__(self):
        super(GooglenetModel128, self).__init__('GooglenetModel128',
                153740040 // 128,
                [346970, 187140, 146661, 106182,
                95118, 84055, 72992, 61929,
                58369, 54810, 51251, 47692,
                46375, 45058, 43741, 42424,
                42077, 41731, 41385, 41039])

class Inception4Model256(Model):
    def __init__(self):
        super(Inception4Model256, self).__init__('Inception4Model256',
                153740040 // 256,
                [5834222, 2988258, 2228434, 1468610,
                1288619, 1108629, 928639, 748649,
                689155, 629661, 570167, 510674,
                488419, 466165, 443911, 421657,
                402481, 383305, 364129, 344953,
                334708, 324464, 314220, 303976,
                296448, 288921, 281394, 273867,
                269776, 265685, 261594, 257504,
                252463, 247423, 242382, 237342,
                234883, 232425, 229966, 227508,
                221494, 215481, 209468, 203455,
                202402, 201350, 200298, 199246,
                195215, 191185, 187155, 183125])

class Resnet50Model128(Model):
    def __init__(self):
        super(Resnet50Model128, self).__init__('Resnet50Model128',
                153740040 // 128,
                [941170, 474975, 367447, 259920, 
                234511, 209103, 183695, 158287,
                147347, 136408, 125468, 114529,
                111929, 109329, 106729, 104129,
                101463, 98797, 96131, 93466,
                92738, 92011, 91283, 90556,
                89939, 89322, 88705, 88089])

class DeepspeechModel64(Model):
    def __init__(self):
        super(DeepspeechModel64, self).__init__('DeepspeechModel64',
                153740040 // 128,
                [619714, 415717, 385933, 356149,
                351970, 347791, 343612, 339434, 
                338022, 336610, 335198, 333786,
                329423, 325060, 320697, 316334,
                314514, 312695, 310875, 309056])

class AutoencoderModel51200(Model):
    def __init__(self):
        super(AutoencoderModel51200, self).__init__('AutoencoderModel51200',
                153740040 // 128,
                [17823579, 9131074, 6901069, 4671065,
                4069033, 3467002, 2864971, 2262940,
                2087242, 1911545, 1735847, 1560150,
                1447719, 1335289, 1222859, 1110429,
                1038625, 966821, 895017, 823213,
                803493, 783773, 764053, 744334,
                725709, 707085, 688461, 669837,
                646499, 623162, 599825, 576488,
                550876, 525265, 499653, 474042,
                460093, 446144, 432195, 418247,
                417658, 417069, 416481, 415891,
                415303, 412098, 408893, 405689,
                392820, 379952, 367083, 354215, 
                350944, 347673, 344402, 341132,
                337458, 333784, 330110, 326437,
                319036, 311635, 304234, 296833])

class TransformerModel4096(Model):
    # Deprecated.
    def __init__(self):
        super(TransformerModel4096, self).__init__('TransformerModel4096',
                153740040 // 128,
                [39936031, 20050013, 14982986, 9915959,
                8689551, 7463143, 6236735, 5010328,
                4588335, 4166343, 3744351, 3322359,
                3114202, 2906046, 2697890, 2489734,
                2367672, 2245611, 2123550, 2001489,
                1921372, 1841255, 1761138, 1681022,
                1622270, 1563518, 1504766, 1446014,
                1401928, 1357843, 1313757, 1269672,
                1232053, 1194434, 1156815, 1119196,
                1088336, 1057477, 1026617, 995758,
                975201, 954644, 934087, 913531,
                896795, 880060, 863324, 846589,
                825506, 804424, 783342, 762260,
                753078, 743896, 734714, 725532,
                710806, 696080, 681354, 666628,
                655069, 643511, 631953, 620395])

class TransformerModel256(Model):
    def __init__(self):
        super(TransformerModel256, self).__init__('TransformerModel256',
                153740040 // 128,
                [2776298, 1466578, 1137556, 808534,
                712710, 616886, 521062, 425236,
                410333, 395430, 380527, 365623,
                344699, 323775, 302851, 281928,
                271382, 260836, 250290, 239742,
                237244, 234746, 232248, 229748,
                228650, 227552, 226454, 225357,
                222478, 219599, 216720, 213839,
                213596, 213353, 213110, 212868,
                209979, 207090, 204201, 201310,
                199848, 198386, 196924, 195461])

class DcganModel256(Model):
    def __init__(self):
        super(DcganModel256, self).__init__('DcganModel256',
                900000000 // 256,
                [282031, 155580, 122417, 89254,
                81969, 74685, 67401, 60117,
                58979, 57841, 56703, 55566,
                53670, 51774, 49878, 47983,
                47107, 46232, 45356, 44481])

class ChatbotModel256(Model):
    def __init__(self):
        super(ChatbotModel256, self).__init__('ChatbotModel256',
                660000000 // 256,
                [114644, 80233, 73326, 66419])

class VideopredictionModel64(Model):
    def __init__(self):
        super(VideopredictionModel64, self).__init__('VideopredictionModel64',
                3200000 // 64,
                [1123488, 625106, 502540, 379974,
                358674, 337375, 316076, 294777,
                286352, 277927, 269502, 261078,
                258117, 255156, 252195, 249234,
                244498, 239762, 235026, 230291, 
                227535, 224779, 222023, 219267,
                219100, 218934, 218768, 218602])

class ToyModel1(Model):
    def __init__(self):
        super(ToyModel1, self).__init__('ToyModel1',
                153740040 // 256,
                [1,1/1.5,1/2.0])

class ToyModel2(Model):
    def __init__(self):
        super(ToyModel2, self).__init__('ToyModel2',
                153740040 // 128,
                [1,1/1.75,1/2.5])

class Models(object):
    """docstring for models"""
    def __init__(self, model_type):
        super(Models, self).__init__()
        
        allowed_model_types = ["realistic", "linear", "toy"]
        if model_type not in allowed_model_types:
            raise ValueError(f"Please choose from {allowed_model_types}")

        self.all_choices = [VggnetModel256(),
            GooglenetModel128(),
            Inception4Model256(),
            Resnet50Model128(),
            DcganModel256(),
            VideopredictionModel64(),
            ChatbotModel256(),
            DeepspeechModel64(),
            TransformerModel256(),
            LinearModel(),
            ToyModel1(),
            ToyModel2()]

        if model_type == "realistic":
            self.choices = [VggnetModel256(),
            GooglenetModel128(),
            Inception4Model256(),
            Resnet50Model128(),
            DcganModel256(),
            VideopredictionModel64(),
            ChatbotModel256(),
            DeepspeechModel64(),
            TransformerModel256()]
        elif model_type == "linear":
            self.choices = [LinearModel()]
        elif model_type == "toy":
            self.choices = [ToyModel1(), ToyModel2()]
        else:
            raise ValueError(f"{model_type} not defined")

        self.max_gpus = max(list(map(lambda m: m.max_gpus, self.choices)))

    def pick_random_model(self, max_gpus):
        

        selected_choices = list(filter(lambda m: m.max_gpus >= max_gpus, self.choices))

        if len(selected_choices) != 0:
            return np.random.choice(selected_choices)
        raise Exception(f"{max_gpus} do not map to any model - currently max gpus are {self.max_gpus}")

    def pick_model_by_name(self, model_name):
        for m in self.all_choices:
            if model_name == m.name:
                return m
        raise ValueError(f"{model_name} not present. Options are: {[m.name for m in self.all_choices]}")