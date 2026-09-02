/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

//
// Created by redamancyguy on 23-5-18.
//
#include <iostream>
#include "../../include/WindowsCompat.h"
#define using_semaphore
#ifdef using_semaphore
sem_t* q_semaphore;
sem_t* pai_semaphore;
#endif
#include "../../include/DEFINE.h"
// #include <c10/cuda/CUDAGuard.h>  // not available in CPU-only build
//#define CB
//#define using_small_network
#include "gen_exp.hpp"

int main(int argc, char const *argv[]) {

#ifdef using_semaphore
    q_semaphore = (sem_t*)mmap(nullptr, sizeof(sem_t), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANON, -1, 0);
    pai_semaphore = (sem_t*)mmap(nullptr, sizeof(sem_t), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANON, -1, 0);
    sem_init(q_semaphore, 1, 1);
    sem_init(pai_semaphore, 1, 1);
#endif
//    double random_rate_discount_rate = 0.997;
    double random_rate_discount_rate = 0.97;
    auto *rs = create_shared_memory<RunningStatus>();
    rs->random_rate = 1;
    std::cout << "rs->random_rate:" << rs->random_rate << std::endl;
    const std::size_t sample_batch = 10;
    // clear_exp(experience_father_path, scanFiles(experience_father_path));  // keep existing files
    rs->exp_num = max_exp_number(scanFiles(experience_father_path)) + 1;
    std::cout << "start_exp:" << rs->exp_num << std::endl;

    // ── Generate initial experience pool (no fork) ──
    const int initial_files = 10;
    std::cout << "Generating " << initial_files << " experience files in main process..." << std::endl;
    for (int i = 0; i < initial_files; i++) {
        gen(rs, 0, sample_batch);
    }
    std::cout << "Experience pool ready." << std::endl;

    break_times = 10;
    auto q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    q_model->to(GPU_DEVICE);
    q_model->train();
    auto q_optimizer = torch::optim::Adam(q_model->parameters(),
                                          torch::optim::AdamOptions(train_lr).
                                                  weight_decay(train_wd));

    std::size_t sample_count = 0;
    while (rs->random_rate > 0) {
        std::cout << "random rate:" << "  " << std::setprecision(30)
                  << rs->random_rate << std::setprecision(6) << std::endl;
        if (sample_count + sample_batch + BATCH_SIZE < count_exp(scanFiles(experience_father_path))) {
            std::cout << "writing shared memory:" << rs->random_rate << std::endl;
            rs->random_rate *= random_rate_discount_rate;
            sample_count += sample_batch;
            training_Q(*q_model, q_optimizer);
            q_model->to(CPU_DEVICE);
            torch::save(q_model, q_model_path);
            q_model->to(GPU_DEVICE);
            std::cout << GREEN << "Q finished:" << sample_count / sample_batch << RESET << std::endl;
            continue;
        }
        // Generate more experience in main process
        std::cout << "Generating more experience..." << std::endl;
        for (int i = 0; i < 10; i++) {
            gen(rs, 0, sample_batch);
        }
    }

#ifdef using_semaphore
    sem_close(q_semaphore);
    sem_destroy(q_semaphore);
    munmap(q_semaphore, sizeof(sem_t));
    q_semaphore = nullptr;
    sem_close(pai_semaphore);
    sem_destroy(pai_semaphore);
    munmap(pai_semaphore, sizeof(sem_t));
    pai_semaphore = nullptr;
#endif
    return 0;
}