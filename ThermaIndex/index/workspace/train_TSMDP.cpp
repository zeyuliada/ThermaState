/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

//
// Created by redamancyguy on 23-8-24.
//

#include <queue>
#include <string>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../include/Parameter.h"
#include "../../include/DataSet.hpp"
#include "../include/Index.hpp"
#include "../include/RL_network.hpp"
//#include "../../include/matplotlibcpp.h"

class ExpSmall{
public:
    float pdf[SMALL_PDF_SIZE]{};
    int dataset_size = 0;
    int action = 0;
//    float memory = 0;
    float reward = 0;
};

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> exp_small_to_tensor(const std::vector<ExpSmall>& exps){
    auto result = std::make_tuple(
            torch::rand({int(exps.size()), SMALL_PDF_SIZE}),
            torch::rand({int(exps.size()), VALUE_SIZE}),
            torch::rand({int(exps.size()), 1}),
            torch::rand({int(exps.size()), 1})
    );
    auto pdf_ptr = std::get<0>(result).data_ptr<float>();
    auto value_ptr = std::get<1>(result).data_ptr<float>();
    auto action_ptr = std::get<2>(result).data_ptr<float>();
    auto reward_ptr = std::get<3>(result).data_ptr<float>();
    for (auto &i:exps) {
//        for(auto j:i.pdf){
//            if(std::isnan(j)){
//                throw MyException("nan value exits !");
//            }
//        }
        std::copy(i.pdf, i.pdf + SMALL_PDF_SIZE, pdf_ptr);
        pdf_ptr += SMALL_PDF_SIZE;
        ///////////////////////////
        value_ptr[0] = float(i.dataset_size);
        value_ptr += VALUE_SIZE;
        ////////////////////////////
        action_ptr[0] = float(i.action);/// i.data_size;
        action_ptr += 1;
        reward_ptr[0] = i.reward;
        reward_ptr += 1;
    }
    return result;
}



std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> exp_small_to_tensor(ExpSmall& exp){
    auto result = std::make_tuple(
            torch::rand({1, SMALL_PDF_SIZE}),
            torch::rand({1, VALUE_SIZE}),
            torch::rand({1, 1}),
            torch::rand({1, 1})
    );
    auto pdf_ptr = std::get<0>(result).data_ptr<float>();
    auto value_ptr = std::get<1>(result).data_ptr<float>();
    auto action_ptr = std::get<2>(result).data_ptr<float>();
    auto reward_ptr = std::get<3>(result).data_ptr<float>();
    auto &i = exp;
    std::copy(i.pdf, i.pdf + SMALL_PDF_SIZE, pdf_ptr);
    pdf_ptr += SMALL_PDF_SIZE;
    ///////////////////////////
    value_ptr[0] = float(i.dataset_size);
    value_ptr += VALUE_SIZE;
    ////////////////////////////
    action_ptr[0] = float(i.action);/// i.data_size;
    action_ptr += 1;
    reward_ptr[0] = i.reward;
    reward_ptr += REWARD_SIZE;
    return result;
}

#define dataset_size_for_tree 50000
//#define dataset_size_for_tree 5000000
#define memory_weight float(0.5)

struct DatasetInfo {
    std::string name;
    int fd;
    size_t num_entries;
};

float random_fanout(){
    return std::pow(2,random_u_0_1() * 8);
}

int heuristic_action(int dataset_size) {
    // 11-action space: 0=Cold, 1=HotRead, 2=HotWrite, 3..10=expand
    if (dataset_size <= DATA_NODE_SIZE) return 0;  // Cold leaf (default)
    else if (dataset_size <= 10)     return 3;  // fanout = 2
    else if (dataset_size <= 50)     return 4;  // fanout = 4
    else if (dataset_size <= 200)    return 5;  // fanout = 8
    else if (dataset_size <= 1000)   return 6;  // fanout = 16
    else if (dataset_size <= 5000)   return 7;  // fanout = 32
    else if (dataset_size <= 20000)  return 8;  // fanout = 64
    else if (dataset_size <= 100000) return 9;  // fanout = 128
    else                             return 10; // fanout = 256
}

std::ofstream log_file("logfile.txt");
int main(){
    std::cout.setf(std::ios::unitbuf);  // disable output buffering
    torch::set_num_threads(1);
    auto q_network = std::make_shared<Small_Q_network>();
    q_network->to(GPU_DEVICE);
    auto q_optimizer = torch::optim::Adam(q_network->parameters());
    q_network->train();
    auto q_target_network = std::make_shared<Small_Q_network>();
    q_target_network->to(GPU_DEVICE);
    q_target_network->eval();
    auto pai_network = std::make_shared<Small_PAI_network>();
    pai_network->to(GPU_DEVICE);
    pai_network->eval();
    float random_rate = 1;
    bool warmup_phase = true;
    const int WARMUP_TREES = 30;
    std::string tmp_pt_path = model_father_path + "tmp.pt";
    if (IsFileExist(tmp_pt_path.c_str())) {
        random_rate = 0.15;
        torch::load(q_network, tmp_pt_path);
        torch::load(q_target_network, tmp_pt_path);
        q_target_network->to(GPU_DEVICE);
        q_target_network->eval();
        warmup_phase = false;
        std::cout << "[Checkpoint] Loaded existing tmp.pt, skipping warm-up. random_rate=" << random_rate << std::endl;
    }
    float discount_rate = 0.9;
    // Lazy-load via pread: ~800KB/slice per tree, works for any file size
    std::vector<DatasetInfo> dataset_infos;
    auto dataset_names = scanFiles(data_father_path);
    for (auto& name : dataset_names) {
        std::string full_path = data_father_path + name;
        int fd = open(full_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "Warning: cannot open " << full_path << std::endl;
            continue;
        }
        off_t file_size = lseek(fd, 0, SEEK_END);
        size_t num_entries = file_size / sizeof(std::pair<KEY_TYPE, VALUE_TYPE>);
        if (num_entries < (size_t)dataset_size_for_tree) {
            close(fd);
            continue;
        }
        dataset_infos.push_back({name, fd, num_entries});
        std::cout << "  [" << name << "] " << num_entries << " entries" << std::endl;
    }
    if (dataset_infos.empty()) {
        std::cerr << "ERROR: No training datasets found in " << data_father_path << std::endl;
        return 1;
    }
    std::cout << "[Data] Opened " << dataset_infos.size() << " datasets (pread-based lazy loading)." << std::endl;

    std::vector<std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>> exp_tensors;
    for(int tree_id = 0;;++tree_id){
        if(tree_id % 33  == 0){
//            shuffle_all_dataset();
        }
        if (warmup_phase) {
            std::cout << "[Warm-up] tree id:" << tree_id << "/" << WARMUP_TREES << std::endl;
        } else {
            std::cout << "tree id:" << tree_id << "  random rate:" << random_rate << std::endl;
        }
        auto tmp_exp_vector = std::vector<std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>>();
        tmp_exp_vector.reserve(exp_tensors.size());
        std::vector<int> shuffle_index;
        shuffle_index.reserve(exp_tensors.size());
        for(int i = 0;i < int(exp_tensors.size());++i){
            shuffle_index.push_back(i);
        }
        std::shuffle(shuffle_index.begin(), shuffle_index.end(),e);
        if(shuffle_index.size() > dataset_size_for_tree * 10){
            shuffle_index.resize(dataset_size_for_tree * 10);
        }
        for(auto i:shuffle_index){
            tmp_exp_vector.push_back(std::move(exp_tensors[i]));
        }
        exp_tensors = std::move(tmp_exp_vector);
        std::unordered_map<int ,int> id_to_father_id;
        std::unordered_map<int ,std::vector<int>> father_id_to_ids;
        std::unordered_map<int ,ExpSmall> id_to_exp;
        std::unordered_map<int ,void *> id_to_node;
        auto& info = dataset_infos[tree_id % dataset_infos.size()];
        auto random_length = std::min(int(dataset_size_for_tree * random_u_0_1_skew(0.2)), int(info.num_entries - 1));
        random_length = std::max(DATA_NODE_SIZE,random_length);
        std::cout << "random_length:"<<random_length<<"  from: "<<info.name << std::endl;
        auto random_start = (int) (e() % (info.num_entries - random_length));
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset(random_length);
        size_t entry_size = sizeof(std::pair<KEY_TYPE, VALUE_TYPE>);
        ssize_t n = pread(info.fd, dataset.data(), random_length * entry_size,
                          random_start * entry_size);
        if (n != ssize_t(random_length * entry_size)) {
            std::cerr << "ERROR: pread short read: " << n << " vs " << (random_length * entry_size) << std::endl;
            throw MyException("pread failed");
        }
        std::sort(dataset.begin(), dataset.end(),
            [](auto& a, auto& b) { return a.first < b.first; });
        // Deduplicate by key (real datasets like wiki may have duplicates)
        auto uniq_end = std::unique(dataset.begin(), dataset.end(),
            [](auto& a, auto& b) { return a.first == b.first; });
        dataset.erase(uniq_end, dataset.end());
        struct Task{
            int node_id;
            int start;
            int stop;
            double lower;
            double upper;
        };
        std::queue<Task> tasks;//data start and end position // lower and upper
        Hits::Index<KEY_TYPE,VALUE_TYPE> *index;
        {
            auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(
                    dataset.begin(),dataset.end());
            index = new Hits::Index<KEY_TYPE,VALUE_TYPE>(Hits::Configuration::default_configuration(),min_max.first,min_max.second);
            index->delete_tree(index->root);
            Hits::InnerNode<KEY_TYPE,VALUE_TYPE>::delete_segment(index->root);
            tasks.push({-1,0,int(dataset.size()),min_max.first,min_max.second});
        }
        int son_id = -1;
        int node_count = 0;
        while(!tasks.empty()){
            if (++node_count % 500 == 0) {
                std::cout << "[progress] processed " << node_count << " nodes, " << tasks.size() << " remaining in queue" << std::endl;
            }
            auto task = tasks.front();
            tasks.pop();
            ExpSmall &exp = id_to_exp[task.node_id];
            auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(
                    dataset.begin() + task.start, dataset.begin() + task.stop,
                    task.lower, task.upper, SMALL_PDF_SIZE);
            std::copy(pdf.begin(),pdf.end(),exp.pdf);
            exp.dataset_size = task.stop - task.start;
            if (warmup_phase) {
                exp.action = heuristic_action(exp.dataset_size);
            } else if(random_u_0_1() > random_rate){
                auto tensor_for_action = exp_small_to_tensor(exp);
                exp.action = 0;
                auto a = std::get<0>(tensor_for_action).mul(torch::ones({int(action_space.size()),std::get<0>(tensor_for_action).size(1)})).to(GPU_DEVICE);
                auto b = std::get<1>(tensor_for_action).mul(torch::ones({int(action_space.size()),std::get<1>(tensor_for_action).size(1)})).to(GPU_DEVICE);
                auto c =   torch::tensor(action_space).view({-1,1}).to(GPU_DEVICE);
                auto pred = q_target_network->forward(a,b,c);
                // Mask invalid actions for small datasets (no expand)
                auto valid = get_valid_actions(exp.dataset_size);
                for(int i = 0;i < (int)action_space.size();++i){
                    bool allowed = false;
                    for(int v:valid) if(action_space[i] == v){ allowed=true; break; }
                    if(!allowed) pred[i] = -std::numeric_limits<float>::infinity();
                }
                pred = torch::softmax(pred,0);
                pred = pred.to(CPU_DEVICE);
                std::vector<float> probability(pred.data_ptr<float>(),pred.data_ptr<float>() + pred.numel());
                auto p = random_u_0_1();
                for(int i = 0;i<int(probability.size());++i){
                    p -= probability[i];
                    if(p < 0){
                        exp.action = int(action_space[i]);
                        break;
                    }
                }
            }else{
                auto valid = get_valid_actions(exp.dataset_size);
                exp.action = valid[e() % valid.size()];
            }
            if(exp.dataset_size == 0){ //zero dataset size
                exp.action = 0;  // Cold leaf
            }
            bool is_leaf_action;
            int action_value;
            decode_action(exp.action, is_leaf_action, action_value);
            if(!is_leaf_action && exp.dataset_size > DATA_NODE_SIZE){
                //split dataset
                //establish a inner node with capacity as fanout  // model based without establish physically
                auto last_memory = Hits::node_memory_count;
                auto inner_node = Hits::InnerNode<KEY_TYPE,VALUE_TYPE>::new_segment(action_value,task.lower,task.upper);
                id_to_node[task.node_id] = inner_node;
//                exp.memory = float(Hits::node_memory_count - last_memory)/ sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::Slot)/ float(std::max(1,exp.dataset_size));
                exp.reward = -QUERY_WEIGHT * inner_cost_weight - MEMORY_WEIGHT * float(Hits::node_memory_count - last_memory) / sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type) / float(std::max(1, exp.dataset_size));
                auto result = Hits::split_dataset<KEY_TYPE,VALUE_TYPE>(dataset.begin() + task.start,dataset.begin() + task.stop, inner_node);
                for(auto const &i:result){
                    tasks.push({++son_id,i.first.first + task.start,i.first.second + task.start,i.second.first,i.second.second});//sub tasks son_id code
                    id_to_father_id[son_id] = task.node_id;
                }
            } else{
                // Leaf type by action value: 0=Cold, 1=HotRead, 2=HotWrite
                auto last_memory = Hits::node_memory_count;
                int leaf_cap = std::max(DATA_NODE_SIZE,int(float(std::max(1,exp.dataset_size)) / default_density));
                void* node_ptr;
                int act_val = exp.action;
                if (act_val == 0) {
                    auto leaf = Hits::ColdLeaf<KEY_TYPE,VALUE_TYPE>::new_segment(leaf_cap,task.lower,task.upper);
                    leaf->bulk_load(dataset.begin() + task.start, dataset.begin() + task.stop);
                    Hits::leaf_cost = 0;
                    for(auto i = task.start;i<task.stop;++i){
                        auto position = leaf->find_with_cost(dataset[i].first);
                        if(position < 0){ std::cerr << "[ColdLeaf find_cost] action="<<exp.action<<" data_size="<<exp.dataset_size <<" key="<<dataset[i].first<<std::endl; throw MyException("bad position!"); }
                    }
                    node_ptr = leaf;
                } else if (act_val == 1) {
                    auto leaf = Hits::HotReadLeaf<KEY_TYPE,VALUE_TYPE>::new_segment(leaf_cap,task.lower,task.upper);
                    for(auto i = task.start;i<task.stop;++i){
                        auto position = leaf->find_insert(dataset[i].first);
                        if(position < 0){ std::cerr << "[HotReadLeaf insert] action="<<exp.action<<" data_size="<<exp.dataset_size <<" leaf_size="<<leaf->size<<" cap="<<leaf->capacity<<" key="<<dataset[i].first<<std::endl; throw MyException("bad position!"); }
                        leaf->array[position] = dataset[i];
                        ++leaf->size;
                    }
                    Hits::leaf_cost = 0;
                    for(auto i = task.start;i<task.stop;++i){
                        auto position = leaf->find_with_cost(dataset[i].first);
                        if(position < 0){ std::cerr << "[HotReadLeaf find_cost] action="<<exp.action<<" data_size="<<exp.dataset_size <<" key="<<dataset[i].first<<std::endl; throw MyException("bad position!"); }
                    }
                    node_ptr = leaf;
                } else if (act_val == 2) {
                    auto leaf = Hits::HotWriteLeaf<KEY_TYPE,VALUE_TYPE>::new_segment(leaf_cap,task.lower,task.upper);
                    for(auto i = task.start;i<task.stop;++i){
                        auto position = leaf->find_insert(dataset[i].first);
                        if(position < 0){ std::cerr << "[HotWriteLeaf insert] action="<<exp.action<<" data_size="<<exp.dataset_size <<" leaf_size="<<leaf->size<<" cap="<<leaf->capacity<<" key="<<dataset[i].first<<std::endl; throw MyException("bad position!"); }
                        leaf->array[position] = dataset[i];
                        ++leaf->size;
                    }
                    Hits::leaf_cost = 0;
                    for(auto i = task.start;i<task.stop;++i){
                        auto position = leaf->find_with_cost(dataset[i].first);
                        if(position < 0){ std::cerr << "[HotWriteLeaf find_cost] action="<<exp.action<<" data_size="<<exp.dataset_size <<" key="<<dataset[i].first<<std::endl; throw MyException("bad position!"); }
                    }
                    node_ptr = leaf;
                } else {
                    std::cerr << "[TSMDP] unexpected action="<<act_val<<" data_size="<<exp.dataset_size<<std::endl;
                    throw MyException("bad action: expand actions should never reach leaf creation");
                }
                id_to_node[task.node_id] = node_ptr;
                exp.reward = -QUERY_WEIGHT * inner_cost_weight - float(double(Hits::leaf_cost) / double(std::max(exp.dataset_size, 1))) * leaf_cost_weight
                             - MEMORY_WEIGHT * float(Hits::node_memory_count - last_memory)/ sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type)/ float(std::max(1,exp.dataset_size));
            }
        }
        for(auto const &i:id_to_father_id){
            father_id_to_ids[i.second].push_back(i.first);
        }
        for(auto &i:father_id_to_ids){
            std::sort(i.second.begin(), i.second.end());
        }
        index->root = (Hits::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[-1];
        if(father_id_to_ids[-1].empty()){
            index->is_leaf = true;
        }
        for(auto const &i:father_id_to_ids){
            bool is_leaf_a; int act_val;
            decode_action(id_to_exp[i.first].action, is_leaf_a, act_val);
            if(i.first != -1 && !is_leaf_a && act_val != int(i.second.size())){
                throw MyException("bad sub ids size !");
            }
            if(i.first != -1 && !is_leaf_a && ((Hits::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[i.first])->capacity != int(i.second.size())){
                throw MyException("bad capacity !");
            }
            int size_count = 0;
            for(auto j:i.second){
                if(id_to_exp.find(j) == id_to_exp.end()){
                    throw MyException("bad finding !");
                }
                size_count += id_to_exp[j].dataset_size;
            }
            if(i.first != -1 && id_to_exp[i.first].dataset_size != size_count){
                throw MyException("un equal dataset size !");
            }
            auto inner_node = (Hits::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[i.first];
            if(i.first != -1 && inner_node->capacity != int(i.second.size())){
                throw MyException("bad size");
            }
            for(int j = 0;j<int(i.second.size());++j) {
                if (father_id_to_ids.find(i.second[j]) != father_id_to_ids.end()) {//this is a inner node
                    set_bitmap(inner_node->bitmap_start(),j);
                    auto sub_node = (Hits::InnerNode<KEY_TYPE, VALUE_TYPE> *) id_to_node[i.second[j]];
                    inner_node->array[j].inner_node = sub_node;
                } else {
                    if (id_to_node.find(i.second[j]) == id_to_node.end()) {
                        throw MyException("not find data node in dict !");
                    }
                    de_set_bitmap(inner_node->bitmap_start(),j);
                    auto data_node = (Hits::DataNode<KEY_TYPE, VALUE_TYPE> *) id_to_node[i.second[j]];
                    inner_node->array[j].data_node = data_node;
                    if (data_node->size != id_to_exp[i.second[j]].dataset_size) {
                        std::cout << data_node->size << ":" << id_to_exp[i.second[j]].dataset_size << std::endl;
                        throw MyException("bad data node size !");
                    }
                }
            }
        }
        Hits::inner_cost = 0;
        Hits::leaf_cost = 0;
        tc.synchronization();
        for(auto i:dataset){
            VALUE_TYPE  value;
            if(!index->get_with_root_leaf(i.first, value) || value != i.second){
                throw MyException("bad get result !");
            }
        }
//        std::cout <<Hits::inner_cost/ double(dataset.size())<< std::endl;
//        std::cout << Hits::leaf_cost/ double(dataset.size())<< std::endl;
//        std::cout <<  tc.get_timer_nanoSec() / double(dataset.size())<< std::endl;

        log_file<<double(Hits::inner_cost) / double(dataset.size())<<","<<double(Hits::leaf_cost) / double(dataset.size())<<","<<tc.get_timer_nanoSec() / double(dataset.size())<<std::endl;
        std::cout <<"dataset:"<<dataset.size()<< std::endl;
        std::cout <<BLUE<<"average inner cost:"<<double(Hits::inner_cost) / double(dataset.size())<<RESET<< std::endl;
        std::cout <<BLUE<<"average leaf cost:"<<double(Hits::leaf_cost) / double(dataset.size())<<RESET<< std::endl;
        std::cout <<BLUE<<"average memory:"<<double(Hits::node_memory_count / sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type)) / double(dataset.size())<<RESET<< std::endl;
        sleep(1);
        delete index;
        {
            for(auto i:id_to_node){//a batch s,a,r,s'
                auto node_id = i.first;//a node id,then we want to get the information of it's sons
                auto u = torch::Tensor();//sons' best target_{t+1}
                if(!warmup_phase && int(father_id_to_ids[node_id].size()) != 0){//this node hash sons
                    std::vector<ExpSmall> son_exps;
                    std::vector<float> exp_dataset_size;
                    for(auto j:father_id_to_ids[node_id]){
                        son_exps.push_back(id_to_exp[j]);
                        exp_dataset_size.push_back(float(son_exps.back().dataset_size));
                    }
                    auto son_tensors = exp_small_to_tensor(son_exps);
                    std::vector<torch::Tensor> tmp_son_result;
                    for(int action = 0;action < int(action_space.size());++action){
                        auto a = std::get<0>(son_tensors).to(GPU_DEVICE);
                        auto b = std::get<1>(son_tensors).to(GPU_DEVICE);
                        auto c = torch::tensor({action}).mul(torch::ones({int(son_exps.size()),1})).to(GPU_DEVICE);
                        auto son_u = q_target_network->forward(a,b,c);
                        tmp_son_result.push_back(son_u.view({son_u.size(0),son_u.size(1),1}));//n 1 1
                    }
                    auto all_son_result = torch::cat(tmp_son_result,2);//n 1 action
                    auto p_for_action = torch::softmax(all_son_result,2);
                    all_son_result = all_son_result.mul(p_for_action).sum(2);
                    all_son_result *= torch::tensor(exp_dataset_size).view({-1,1}).to(GPU_DEVICE);
                    all_son_result /= float(id_to_exp[node_id].dataset_size);
                    all_son_result = all_son_result.sum(0);
                    u = all_son_result.view({-1,1}).detach().to(CPU_DEVICE);
                }else{
                    u = torch::zeros({1,1});
                }
                auto tensor_exp = exp_small_to_tensor(id_to_exp[node_id]);
                exp_tensors.emplace_back(std::get<0>(tensor_exp),std::get<1>(tensor_exp),std::get<2>(tensor_exp),std::get<3>(tensor_exp),u);
            }
        }
        q_network->to(GPU_DEVICE);
        float loss_count = 0;

        for(int k = 0;k < 100 ;++k){
            std::vector<torch::Tensor> pdf_batch;
            std::vector<torch::Tensor> value_batch;
            std::vector<torch::Tensor> action_batch;
            std::vector<torch::Tensor> reward_batch;
            std::vector<torch::Tensor> u_batch;
            for(int i = 0;i<BATCH_SIZE;++i){
                auto &draw_sample = exp_tensors[e() % exp_tensors.size()];
                pdf_batch.push_back(std::get<0>(draw_sample));
                value_batch.push_back(std::get<1>(draw_sample));
                action_batch.push_back(std::get<2>(draw_sample));
                reward_batch.push_back(std::get<3>(draw_sample));
                u_batch.push_back(std::get<4>(draw_sample));
            }
            auto a_batch = torch::vstack(pdf_batch).detach().to(GPU_DEVICE);
            auto b_batch = torch::vstack(value_batch).detach().to(GPU_DEVICE);
            auto c_batch = torch::vstack(action_batch).detach().to(GPU_DEVICE);
            auto d_batch = torch::vstack(reward_batch).detach().to(GPU_DEVICE);
            auto e_batch = torch::vstack(u_batch).detach().to(GPU_DEVICE);
            auto pred = q_network->forward(a_batch,b_batch ,c_batch);
            auto target = d_batch +  e_batch * discount_rate;
            auto loss = torch::nn::L1Loss()->forward(pred,target);
            q_optimizer.zero_grad();
            loss.backward();
            q_optimizer.step();
            loss_count += loss.item().toFloat();
        }
        std::cout <<RED<<"loss_count:"<<loss_count / float(100)<<"  nodes:"<<id_to_node.size()<<RESET<< std::endl;
        if (warmup_phase && tree_id >= WARMUP_TREES - 1) {
            std::cout << GREEN << "[Warm-up] Intensive pretraining (500 steps)..." << RESET << std::endl;
            for (int k = 0; k < 500; ++k) {
                std::vector<torch::Tensor> pdf_batch, value_batch, action_batch, reward_batch, u_batch;
                for (int i = 0; i < BATCH_SIZE; ++i) {
                    auto &draw_sample = exp_tensors[e() % exp_tensors.size()];
                    pdf_batch.push_back(std::get<0>(draw_sample));
                    value_batch.push_back(std::get<1>(draw_sample));
                    action_batch.push_back(std::get<2>(draw_sample));
                    reward_batch.push_back(std::get<3>(draw_sample));
                    u_batch.push_back(std::get<4>(draw_sample));
                }
                auto a_batch = torch::vstack(pdf_batch).detach().to(GPU_DEVICE);
                auto b_batch = torch::vstack(value_batch).detach().to(GPU_DEVICE);
                auto c_batch = torch::vstack(action_batch).detach().to(GPU_DEVICE);
                auto d_batch = torch::vstack(reward_batch).detach().to(GPU_DEVICE);
                auto pred = q_network->forward(a_batch, b_batch, c_batch);
                auto loss = torch::nn::L1Loss()->forward(pred, d_batch);
                q_optimizer.zero_grad();
                loss.backward();
                q_optimizer.step();
            }
            std::cout << GREEN << "[Warm-up] Complete. Saving model to " << tmp_pt_path << RESET << std::endl;
            q_network->to(CPU_DEVICE);
            torch::save(q_network, tmp_pt_path);
            std::cout << GREEN << "[Warm-up] tmp.pt saved. Done." << RESET << std::endl;
            for (auto& info : dataset_infos) close(info.fd);
            return 0;
        }
        if(!warmup_phase && tree_id > 3){
            int update_frequency = 10;
            if(tree_id % update_frequency == update_frequency-1){
                std::cout <<GREEN<<"update q network "<<RESET<< std::endl;
                q_network->to(CPU_DEVICE);
                torch::save(q_network,model_father_path+"tmp.pt");
                torch::load(q_target_network,model_father_path+"tmp.pt");
                q_target_network->to(GPU_DEVICE);
                q_network->to(GPU_DEVICE);
            }
            random_rate *= 0.9995;
//            random_rate *= 0.995;
        }
        if(random_rate < 3e-3){
            break;
        }
    }
    for (auto& info : dataset_infos) close(info.fd);
}
