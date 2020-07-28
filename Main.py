import sys

sys.path.append('../../')
import numpy as np
import Lock
import time
import os
from os.path import join
from datetime import datetime
from util import utilities as Utils

from core4.AspectLearnerGSOM import AspectLearnerGSOM
from core4.AssociativeGSOM import AssociativeGSOM
from util import display as Display_Utils
from params import params as Params


def generate_output_config(SF, forget_threshold):
    # File Config
    dataset = 'Classifier'
    experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('output/', experiment_id)

    # Output data config
    output_save_filename = '{}_data_'.format(dataset)
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
        forget_threshold) + 'itr'
    plot_output_name = join(output_save_location, filename)

    # Generate output plot location
    output_loc = plot_output_name
    output_loc_images = join(output_loc, 'images/')
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(output_loc_images):
        os.makedirs(output_loc_images)

    return output_loc, output_loc_images


def dispaly(gsom_nodemap, classes, name):
    # Display
    display = Display_Utils.Display(gsom_nodemap, None)
    display.setup_labels_for_gsom_nodemap(classes, 2, name + ' Latent Space of {} : SF={}'.format("Data", SF),
                                          join(output_loc, name + ' latent_space_' + str(SF) + '_hitvalues'))
    display.setup_labels_for_gsom_nodemap(classes, 2, name + ' Latent Space of {} : SF={}'.format("Data", SF),
                                          join(output_loc, name + ' latent_space_' + str(SF) + '_labels'))
    print(name + ' Plotting Completed. \n')


if __name__ == "__main__":
    SF = 0.83
    forget_threshold = 60
    temporal_contexts = 1
    learning_itr = 100
    smoothing_irt = 50
    plot_for_itr = 4

    # Init GSOM Parameters
    gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt,
                                        distance=Params.DistanceFunction.EUCLIDEAN,
                                        temporal_context_count=temporal_contexts,
                                        forget_itr_count=forget_threshold)
    generalise_params = Params.GeneraliseParameters(gsom_params)

    # Setup the age threshold based on the input vector length
    generalise_params.setup_age_threshold(Lock.INPUT_SIZE)

    # Process the input files
    output_loc, output_loc_images = generate_output_config(SF, forget_threshold)

    X_train_emotion = Lock.emotion_feature
    y_train_emotion = Lock.emotion_label

    X_train_behaviour = Lock.behaviour_feature
    y_train_behaviour = Lock.behaviour_label

    y_train_threat = Lock.threat_label

    print("Emotion feature is ", X_train_emotion.shape)
    print("Emotion label is ", y_train_emotion.shape)

    print("Behavior feature is ", X_train_behaviour.shape)
    print("Behavior label is ", y_train_behaviour.shape)

    print("Thread level label is ", y_train_threat.shape)

    emotion_result_dict = []
    behavior_result_dict = []
    threat_result_dict = []

    final_emotion_result_dict = []
    final_behavior_result_dict = []
    final_threat_result_dict = []

    start_time = time.time()

    EmotionGSOM = AspectLearnerGSOM(generalise_params.get_gsom_parameters(), "emotion", X_train_emotion,
                                    X_train_emotion.shape[1],
                                    plot_for_itr=plot_for_itr,
                                    activity_classes=y_train_emotion, output_loc=output_loc_images)

    BehaviourGSOM = AspectLearnerGSOM(generalise_params.get_gsom_parameters(), "behaviour", X_train_behaviour,
                                      X_train_behaviour.shape[1],
                                      plot_for_itr=plot_for_itr,
                                      activity_classes=y_train_behaviour, output_loc=output_loc_images)

    ThreatGSOM = AssociativeGSOM(generalise_params.get_gsom_parameters(),
                                 X_train_emotion.shape[1] + X_train_behaviour.shape[1],
                                 plot_for_itr=plot_for_itr,
                                 activity_classes=y_train_behaviour, output_loc=output_loc_images)

    EmotionGSOM.start()
    BehaviourGSOM.start()
    ThreatGSOM.start()

    EmotionGSOM.join()
    BehaviourGSOM.join()
    ThreatGSOM.join()

    print("Plotting Emotion Nodemap")
    dispaly(EmotionGSOM.gsom_nodemap, y_train_emotion, name="Emotion")

    print("Plotting Behaviour Nodemap")
    dispaly(BehaviourGSOM.gsom_nodemap, y_train_behaviour, name="Behaviour")

    print("Plotting Threat Nodemap")
    dispaly(ThreatGSOM.gsom_nodemap, y_train_threat, name="Threat")

    print("Saving Emotion Nodemap")
    emotion_result_dict.append({
        'gsom': EmotionGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(emotion_result_dict,
                                join(EmotionGSOM.output_save_location, 'emotion-gsom_nodemap_SF-{}'.format(SF)))

    print("Saving Behavior Nodemap")
    behavior_result_dict.append({
        'gsom': BehaviourGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(behavior_result_dict,
                                join(BehaviourGSOM.output_save_location, 'behavior-gsom_nodemap_SF-{}'.format(SF)))

    print("Saving Threat Nodemap")
    threat_result_dict.append({
        'gsom': ThreatGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(threat_result_dict,
                                join(ThreatGSOM.output_save_location, 'threat-gsom_nodemap_SF-{}'.format(SF)))

    print("Saving finalized emotion nodemap")
    EmotionGSOM.finalize_gsom_label()
    final_emotion_result_dict.append({
        'gsom': EmotionGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(final_emotion_result_dict,
                                join(EmotionGSOM.output_save_location, 'final-emotion-gsom_nodemap_SF-{}'.format(SF)))

    print("Saving finalized behavior nodemap")
    BehaviourGSOM.finalize_gsom_label()
    final_behavior_result_dict.append({
        'gsom': BehaviourGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(final_behavior_result_dict,
                                join(BehaviourGSOM.output_save_location, 'final-behavior-gsom_nodemap_SF-{}'.format(SF)))

    print("Saving finalized threat nodemap")
    ThreatGSOM.finalize_gsom_label()
    final_threat_result_dict.append({
        'gsom': ThreatGSOM.gsom_nodemap,
        'aggregated': None
    })
    Utils.Utilities.save_object(final_threat_result_dict,
                                join(ThreatGSOM.output_save_location, 'final-threat-gsom_nodemap_SF-{}'.format(SF)))



    # label = np.load("data/behavior_label.npy")
    # emotion_label = np.where(label==5, 0, label)
    # emotion_label = np.where(label==6, 5, label)
    # emotion_label = np.where(label==7, 5, label)
    #
    # threat_label = np.where(label == 1, 9, label)
    # threat_label = np.where(label == 4, 1, label)
    # threat_label = np.where(label == 2, 4, label)
    # threat_label = np.where(label == 3, 2, label)
    # threat_label = np.where(label == 9, 3, label)
    # threat_label = np.where(label == 6, 3, label)
    # threat_label = np.where(label == 7, 3, label)
    # threat_label = np.where(label == 0, 5, label)
    #
    # np.save("data/emotion_label.npy", emotion_label)
    # np.save("data/threat_label.npy", threat_label)