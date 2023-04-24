import random

class precision_recall_calculator:
    
    def __init__(self, test_data, train_data, pm, user_model, item_model):
        self.test_data = test_data
        self.train_data = train_data
        self.user_test_sample = None
        self.model1 = pm
        self.model2 = user_model
        self.model3 = item_model
        users = test_data['user_id'].unique()
        self.id_to_no = { id:no for no,id in enumerate(users) }

        self.isi_training_dict = {}
        self.isu_training_dict = {}
        self.pm_training_dict = {}
        self.isui_training_dict = {}
        self.test_dict = {}
        self.songs_to_rats_u = {}
        self.songs_to_rats_i = {}
        self.songs_to_rats_ui = {}
    
    #Method to return random percentage of values from a list
    def remove_percentage(self, list_a, percentage):
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
    
        return new_list
    
    #Create a test sample of users for use in calculating precision
    #and recall
    def create_user_test_sample(self, percentage):
        #Find users common between training and test set
        users_test_and_training = list(set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))
        print("Length of user_test_and_training:%d" % len(users_test_and_training))

        #Take only random user_sample of users for evaluations
        self.users_test_sample = self.remove_percentage(users_test_and_training, percentage)

        print("Length of user sample:%d" % len(self.users_test_sample))
        
    #Method to generate recommendations for users in the user test sample
    def get_test_sample_recommendations(self):
        #For these test_sample users, get top 10 recommendations from training set
        

        for user_id in self.users_test_sample:
            #Get items for user_id from user similarity model
            print("Getting recommendations for user:%s" % user_id)

            no = 20

            user_sim_users,u_rats = self.model2(user_id,no)
            self.songs_to_rats_u[user_id] = { user_sim_users[i] : u_rats[i] for i in range(len(user_sim_users)) }

            self.isu_training_dict[user_id] = list(user_sim_users)

            user_sim_items,i_rats = self.model3(user_id,no)
            self.isi_training_dict[user_id] = list(user_sim_items)
            self.songs_to_rats_i[user_id] = { user_sim_items[i] : i_rats[i] for i in range(len(user_sim_items)) }

            ui_songs = user_sim_items[:int(len(user_sim_items)/2)+1] + user_sim_users[:int(len(user_sim_users)/2)+1]
            ui_rats = u_rats[:int(len(user_sim_items)/2)+1] + i_rats[:int(len(user_sim_users)/2)+1]

            self.isui_training_dict[user_id] = user_sim_items[:int(len(user_sim_items)/2)+1] + user_sim_users[:int(len(user_sim_users)/2)+1]
            self.songs_to_rats_ui[user_id] = { ui_songs[i] : ui_rats[i] for i in range(len(ui_songs)) }


            #Get items for user_id from popularity model
            user_sim_items = self.model1(user_id,no)
            self.pm_training_dict[user_id] = list(user_sim_items)

            
    
            #Get items for user_id from test_data
            test_data_user = self.test_data[self.test_data['user_id'] == user_id]
            self.test_dict[user_id] = set(test_data_user['song'].unique() )
    
    #Method to calculate the precision and recall measures
    def calculate_precision_recall(self):
        #Create cutoff list for precision and recall calculation
        cutoff_list = list(range(1,21))


        #For each distinct cutoff:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.

        isu_avg_precision_list = []
        isu_avg_recall_list = []
        isi_avg_precision_list = []
        isi_avg_recall_list = []
        isui_avg_precision_list = []
        isui_avg_recall_list = []
        pm_avg_precision_list = []
        pm_avg_recall_list = []

        u_mae = []
        i_mae = []
        ui_mae = []


        num_users_sample = len(self.users_test_sample)
        for N in cutoff_list:
            isu_sum_precision = 0
            isu_sum_recall = 0
            pm_sum_precision = 0
            pm_sum_recall = 0
            isi_sum_precision = 0
            isi_sum_recall = 0
            isi_avg_precision = 0
            isi_avg_recall = 0
            isu_avg_precision = 0
            isu_avg_recall = 0
            isui_avg_precision = 0
            isui_avg_recall = 0
            isui_sum_precision = 0
            isui_sum_recall = 0
            pm_avg_precision = 0
            pm_avg_recall = 0
            u_sum_mae = 0
            i_sum_mae = 0
            ui_sum_mae = 0
            u_avg_mae = 0
            i_avg_mae = 0
            ui_avg_mae = 0


            for user_id in self.users_test_sample:
                isu_hitset = self.test_dict[user_id].intersection(set(self.isu_training_dict[user_id][0:N]))
                isi_hitset = self.test_dict[user_id].intersection(set(self.isi_training_dict[user_id][0:N]))
                isui_hitset = self.test_dict[user_id].intersection(set(self.isui_training_dict[user_id][0:N]))
                pm_hitset = self.test_dict[user_id].intersection(set(self.pm_training_dict[user_id][0:N]))
                testset = self.test_dict[user_id]
        
                pm_sum_precision += float(len(pm_hitset))/float(N)
                pm_sum_recall += float(len(pm_hitset))/float(len(testset))

                isu_sum_recall += float(len(isu_hitset))/float(len(testset))
                isu_sum_precision += float(len(isu_hitset))/float(N)

                isui_sum_recall += float(len(isui_hitset))/float(len(testset))
                isui_sum_precision += float(len(isui_hitset))/float(N)

                isi_sum_recall += float(len(isi_hitset))/float(len(testset))
                isi_sum_precision += float(len(isi_hitset))/float(N)
            
                for i in isu_hitset:
                    u_sum_mae+=abs(self.songs_to_rats_u[user_id][i]-self.test_data.iloc[self.id_to_no[user_id],8])
                if len(isu_hitset) != 0:
                    u_sum_mae= u_sum_mae/ len(isu_hitset)


                for i in isi_hitset:
                    i_sum_mae+=abs(self.songs_to_rats_i[user_id][i]-self.test_data.iloc[self.id_to_no[user_id],8])
                if len(isi_hitset) != 0:
                    i_sum_mae= i_sum_mae/len(isi_hitset)

                for i in isui_hitset:
                    ui_sum_mae+=abs(self.songs_to_rats_ui[user_id][i]-self.test_data.iloc[self.id_to_no[user_id],8])
                if len(isui_hitset) != 0:
                    ui_sum_mae= ui_sum_mae/ len(isui_hitset)
                
                

            pm_avg_precision = pm_sum_precision/float(num_users_sample)
            pm_avg_recall = pm_sum_recall/float(num_users_sample)
    
            isu_avg_precision = isu_sum_precision/float(num_users_sample)
            isu_avg_recall = isu_sum_recall/float(num_users_sample)

            isui_avg_precision = isui_sum_precision/float(num_users_sample)
            isui_avg_recall = isui_sum_recall/float(num_users_sample)

            isu_avg_precision_list.append(isu_avg_precision)
            isu_avg_recall_list.append(isu_avg_recall)

            isi_avg_precision = isi_sum_precision/float(num_users_sample)
            isi_avg_recall = isi_sum_recall/float(num_users_sample)

            u_avg_mae= u_sum_mae/ float(num_users_sample)
    
            i_avg_mae= i_sum_mae /float(num_users_sample)
            ui_avg_mae = ui_sum_mae /float(num_users_sample)

            u_mae.append(u_avg_mae)
            i_mae.append(i_avg_mae)
            ui_mae.append(ui_avg_mae)

            isi_avg_precision_list.append(isi_avg_precision)
            isi_avg_recall_list.append(isi_avg_recall)

            isui_avg_precision_list.append(isui_avg_precision)
            isui_avg_recall_list.append(isui_avg_recall)

            pm_avg_precision_list.append(pm_avg_precision)
            pm_avg_recall_list.append(pm_avg_recall)
            
        return (pm_avg_precision_list, pm_avg_recall_list, isu_avg_precision_list, isu_avg_recall_list, isi_avg_precision_list, isi_avg_recall_list, isui_avg_precision_list, isui_avg_recall_list,u_mae,i_mae,ui_mae)
     

    #A wrapper method to calculate all the evaluation measures
    def calculate_measures(self, percentage):
        #Create a test sample of users
        self.create_user_test_sample(percentage)
        
        #Generate recommendations for the test sample users
        self.get_test_sample_recommendations()
        
        #Calculate precision and recall at different cutoff values
        #for popularity mode (pm) as well as item similarity model (ism)
        
        return self.calculate_precision_recall()
    #return (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) 



