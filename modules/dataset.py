import os
import operator

import numpy as np

from modules.utils import *

def select_labels_range_from_dict(dictionary, labels, begin, end):
    result = {}
    for i in range(begin, end):
        result[list(labels)[i]] = dictionary[list(labels)[i]]
    return result

def dict_to_numpy_array(dictionary):
    arr = np.array([value for (_, value) in dictionary.items()])
    return arr

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

#Represents a single MMPI2 interview
class MMPI2InterviewSample:
    def __init__(self,
                personal_info,
                psychometric_scales,
                survey_answers
                ):


        self.extracted_from_sample = int(personal_info["Campione"])
        self.mmpi_id = int(personal_info["CodiceMMPI"])
        self.couple_id = int(personal_info["Ncoppia"])
        self.gender = int(personal_info["Sesso"])
        self.age = int(personal_info["Età"])

        #"Scolarità" anni di scolarità
        self.scolarship_age = int(personal_info["Scolarita"])

        self.marital_status = personal_info["StatoCivile"]
        self.profession = personal_info["Professione"]
        self.psychiatric = False if personal_info["Psichiatrici"] == "No" else True
        
        self.citizenship = personal_info["Cittadinanza"]

        #PMA : procreazione medicalmente assistita
        self.assisted_procreation = False if personal_info["PMA"] == "No" else True
        
        #Same data as before but as a dictionary for readiness of use
        self.personal_info = {
            "extracted_from_sample" : self.extracted_from_sample,
            "mmpi_id" : self.mmpi_id,
            "couple_id" : self.couple_id,
            "gender" : self.gender,
            "age" : self.age,
            "scolarship_age" : self.scolarship_age,
            "marital_status" : self.marital_status,
            "profession" : self.profession,
            "psychiatric" : self.psychiatric,
            "citizenship" : self.citizenship,
            "assisted_procreation" : self.assisted_procreation
        } 
        #print(self.personal_info)
        
        #Convert all psych values to float
        for psy_key, psy_value in psychometric_scales.items():
            psychometric_scales[psy_key] = float(psy_value)
        
        self.psychometric_scales = psychometric_scales.copy()
        self.psychometric_scales_arr = [value for (_, value) in self.psychometric_scales.items()]

        self.psychometric_values = {}

        #Validity scales: determine if patients are non-responding or inconsistent responding (CNS, VRIN, TRIN), overreporting or exaggerating symptoms (F, Fb, Fp, FBS)
        #or under-reporting/downplaying psychological symptoms (L, K, S)
        #print(list(psychometric_scales.keys())[0:16])
        self.psychometric_values["validity"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 0, 16)
                
        #Clinical scales: diagnoses of various psychiatric conditions (Hypocondriasis ...)
        #print(list(psychometric_scales.keys())[16:102])
        self.psychometric_values["clinical"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 16, 102)
        
        #Content scales: Used to measure symptoms of psychiatric conditions (Anxiety, obsessivenes, ...) so they need to be read in addition to clinical scales
        #print(list(psychometric_scales.keys())[102:186])
        self.psychometric_values["content"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 102, 186)
        
        #Supplemental scales: used in supplement to the content scales to determine if some of the symptoms are caused by different additional possible causes ("controlled hostility", alcoholism ...)
        #print(list(psychometric_scales.keys())[186:210])
        self.psychometric_values["supplemental"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 186, 210)

        #Psy-5 scales measure dimensional traits of personality disorders
        #print(list(psychometric_scales.keys())[228:238])
        self.psychometric_values["psy5"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 218, 228)

        #undefined: probably also Psy-5
        #print(list(psychometric_scales.keys())[210:228])
        self.psychometric_values["other"] = select_labels_range_from_dict(psychometric_scales, psychometric_scales.keys(), 210, 218)
        
        self.survey_answers = {}
        for survey_item, item_answer in survey_answers.items():
            #self.survey_answers[survey_item] = True if item_answer == '1' else False
            self.survey_answers[survey_item] = int(item_answer)
        
        #print(self.survey_answers)
        
    #Represent only relevant values as a string
    def __str__(self):
        res = "\n\nExtracted from population sample (ID): "+str(self.extracted_from_sample)
        res+= " (couple ID: "+str(self.couple_id)+")\n"
        res+= "MMPI ID: "+str(self.mmpi_id)+"\n"
        res+= "Age: "+str(self.age)+"\n"
        res+= "Gender: "+str(self.gender)+"\n\n"

        res+= "Instruction level: "+str(self.scolarship_age)+"\n"
        res+= "Marital status: "+str(self.marital_status)+"\n"
        res+= "Profession: "+str(self.profession)+"\n"
        res+= "Citizenship: "+str(self.citizenship)+"\n"
        res+= "Assisted procreation: "+str(self.assisted_procreation)+"\n\n\n"
        res+= "Psychometric values: \n"
        res+= "\tValidity:\n"+str(self.psychometric_values["validity"])+"\n"
        res+= "\tClinical:\n"+str(self.psychometric_values["clinical"])+"\n"
        res+= "\tValidity:\n"+str(self.psychometric_values["content"])+"\n"
        res+= "\tValidity:\n"+str(self.psychometric_values["supplemental"])+"\n"
        res+= "\tValidity:\n"+str(self.psychometric_values["psy5"])+"\n"
        res+= "\tValidity:\n"+str(self.psychometric_values["other"])+"\n"
        res+= "Survey: "+str(self.survey_answers)+"\n\n"

        return res

    def _sample_to_dict(self):
        return {**self.personal_info, **self.psychometric_values, **self.survey_answers}

    #String representing all the psychometric values as a dictionary
    def __repr__(self):
        return str(self._sample_to_dict())
    
    #Get value of a specific psychometric value ("key")
    def __getitem__(self, key):
        if key in self.personal_info.keys():
            return self.personal_info[key]
        elif key in self.psychometric_values.keys():
            return self.psychometric_values[key]
        elif key in self.survey_answers.keys():
            return self.survey_answers[key]
        else:
            raise KeyError()
    
    #Psychometric evaluation (scales divided by groups)
    def psy_scale_groups_to_numpy_array(self, scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]):
        all_scales =  {}
        for scale_class in scale_classes:
            if scale_class not in self.psychometric_values.keys():
                raise KeyError("Scale class not found")

            all_scales = merge_dicts(all_scales, self.psychometric_values[scale_class])
        all_scales = dict_to_numpy_array(all_scales)
        return all_scales

    #Psychometric evaluation (single scales)
    def psy_scales_to_numpy_array(self, scale_indices):
        selected_scales = []
        for idx in scale_indices:
            selected_scales.append(self.psychometric_scales_arr[idx])
            
        return np.array(selected_scales)

    #Boolean survey
    def survey_to_numpy_array(self):
        return dict_to_numpy_array(self.survey_answers)


class MMPI2Dataset:

    #Creates an instance of the Dataset: if "samples" is None, creates an empty instance
    def __init__(self, labels, samples = None) -> None:
        self.samples = []
        self.couples_to_samples = {}
        if samples is not None:
            self.samples = samples
            
            #Build couple index
            self.couples_to_samples = self.build_couple_index(self.samples)
        else:
            self.couples_to_samples = {}
        
        self.labels = labels
        self.psy_scales_labels = labels[11:246]
        #Create the label vocabulary
        label_to_idx, idx_to_label = create_indices(labels)

        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label

    #Adds the couple component to the couple index.
    #Each couple is given a couple_id: 
    # - if a match is found in the couples_to_samples index, return an updated tuple with the previous component and the new component
    # - otherwise return add tuple to the couple index with only the first sample (the second is left to None)
    def update_couple_index(self, sample):
        if sample["couple_id"] in self.couples_to_samples:
            self.couples_to_samples[sample["couple_id"]] = (self.couples_to_samples[sample["couple_id"]][0], sample)
        else:
            self.couples_to_samples[sample["couple_id"]] = (sample, None)
    
    #Associates samples in couples based on couple_id
    def build_couple_index(self, samples):
        for sample in self.samples:
            self.update_couple_index(sample)

    #Create a sample instance and add it to this dataset instance and couples it with the other sample in the couple or to a new couple instance
    def add_sample(self, personal_info, psy_scales, survey_answers):
        sample = MMPI2InterviewSample(personal_info, psy_scales, survey_answers)
        self.samples.append(sample)
        self.update_couple_index(sample)

    #Loads a dataset from a file and returns it as an instance of this class
    @classmethod
    def load_dataset(cls, dataset_dir, dataset_filename):
        data, labels = load_data_from_csv(dataset_dir, dataset_filename)

        personal_info_labels = labels[:11]
        psy_scales_labels = labels[11:246]
        bool_answers_labels = labels[246:]
                
        dataset = MMPI2Dataset(labels)

        #Initialize dataset as a list of objects
        for sample in data:
            personal_info = merge_labels_and_values_as_dict(personal_info_labels, sample[:11])
            psy_scales = merge_labels_and_values_as_dict(psy_scales_labels, sample[11:246])
            bool_answers = merge_labels_and_values_as_dict(bool_answers_labels, sample[246:])
            #dataset.append(MMPI2InterviewSample(personal_info, psy_scales, bool_answers))      
            dataset.add_sample(personal_info, psy_scales, bool_answers)
            
        return dataset

    def __repr__(self):
        return [sample.__repr__() for sample in self.samples]

    def __str__(self):
        return "\n\nMMPI-2 Dataset with "+str(len(self.samples))+" entries\n"+str(self.__repr__())

    def __get__(self, index):
        return self.sample[index]

##### SCALE GROUPS #####    

    #Converts only the psychometric values from the selected classes for all samples to a numpy array (2D tensor)
    def psy_scale_groups_to_numpy_array(self, scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]):
        arr = []
        for sample in self.samples:
            arr.append(sample.psy_scale_groups_to_numpy_array(scale_classes))

        return np.stack(arr)

    #Returns a numpy array (2D tensor) for all the first components of a couple and one for the second components of a couple
    def psy_scale_groups_aligned_couple_arrays(self, scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]):
        arr_first_elements_of_couples = []
        arr_second_elements_of_couples = []

        for (couple_id, couple_components) in self.couples_to_samples.items():

            arr_first_elements_of_couples.append(couple_components[0].psy_scale_groups_to_numpy_array(scale_classes))
            arr_second_elements_of_couples.append(couple_components[1].psy_scale_groups_to_numpy_array(scale_classes))

        return np.stack(arr_first_elements_of_couples), np.stack(arr_second_elements_of_couples)

    #Computes for each couple the difference between psychometric values of the selected classes
    def psy_scale_groups_couple_difference_to_numpy_array(self, scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]):
        np_arr_first_elements_of_couples, np_arr_second_elements_of_couples = self.psy_scale_groups_aligned_couple_arrays(scale_classes = scale_classes)

        return np_arr_first_elements_of_couples - np_arr_second_elements_of_couples
        
    def binary_array_to_psy_scale_groups_to_numpy_array(self, psy_scale_groups_activation):
        scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]
        assert len(psy_scale_groups_activation) == len(scale_classes), "Expected: "+str(len(scale_classes))+", Received: "+str(len(psy_scale_groups_activation))

        returned_scale_groups = []
        for i, scale_group in enumerate(scale_classes):
            if psy_scale_groups_activation[i]>0: 
                returned_scale_groups.append(scale_group)
        return self.psy_scale_groups_to_numpy_array(scale_classes = returned_scale_groups)

    def binary_array_to_psy_scale_groups_couple_difference_to_numpy_array(self, psy_scale_groups_activation):
        scale_classes = ["validity", "clinical", "content", "supplemental", "psy5", "other"]
        assert len(psy_scale_groups_activation) == len(scale_classes), "Expected: "+str(len(scale_classes))+", Received: "+str(len(psy_scale_groups_activation))

        returned_scale_groups = []
        for i, scale_group in enumerate(scale_classes):
            if psy_scale_groups_activation[i]>0: 
                returned_scale_groups.append(scale_group)
        return self.psy_scale_groups_couple_difference_to_numpy_array(scale_classes = returned_scale_groups)

########################

    def get_psychometric_scales_indices(self):
        return range(11,246)
    def get_psychometric_scales_groups_indices(self):
        return range(len(["validity", "clinical", "content", "supplemental", "psy5", "other"]))

##### Single scales #####
    #Converts only the psychometric values from the selected classes for all samples to a numpy array (2D tensor)
    def psy_scales_to_numpy_array(self, scale_activations = None):
        if scale_activations is None:
            scale_indices = self.get_psychometric_scales_indices()
        else:
            scale_indices = np.nonzero(scale_activations)[0]

        arr = []
        for sample in self.samples:
            arr.append(sample.psy_scales_to_numpy_array(scale_indices))

        return np.stack(arr)

    #Returns a numpy array (2D tensor) for all the first components of a couple and one for the second components of a couple
    def psy_scales_aligned_couple_arrays(self, scale_indices = None):
        if scale_indices is None:
            scale_indices = self.get_psychometric_scales_indices()

        arr_first_elements_of_couples = []
        arr_second_elements_of_couples = []

        for (couple_id, couple_components) in self.couples_to_samples.items():

            arr_first_elements_of_couples.append(couple_components[0].psy_scales_to_numpy_array(scale_indices))
            arr_second_elements_of_couples.append(couple_components[1].psy_scales_to_numpy_array(scale_indices))

        return np.stack(arr_first_elements_of_couples), np.stack(arr_second_elements_of_couples)

    #Computes for each couple the difference between psychometric values of the selected classes
    def psy_scales_couple_difference_to_numpy_array(self, scale_indices = None):
        if scale_indices is None:
            scale_indices = self.get_psychometric_scales_indices()

        np_arr_first_elements_of_couples, np_arr_second_elements_of_couples = self.psy_scales_aligned_couple_arrays(scale_indices)

        return np_arr_first_elements_of_couples - np_arr_second_elements_of_couples
        
    def binary_array_to_psy_scales_to_numpy_array(self, psy_scales_activation):
        assert len(psy_scales_activation) == 235, "An activation value 1/0 has to be specified for each psychometric scale"
        return self.psy_scales_to_numpy_array(psy_scales_activation)

    def binary_array_to_psy_scales_couple_difference_to_numpy_array(self, psy_scales_activation):
        assert len(psy_scales_activation) == 235, "An activation value 1/0 has to be specified for each psychometric scale"
        return self.psy_scales_couple_difference_to_numpy_array(psy_scales_activation)
        
#########################

    def survey_to_numpy_array(self):
        arr = []
        for sample in self.samples:
            arr.append(sample.survey_to_numpy_array())
        return np.stack(arr)