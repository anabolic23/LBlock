#include "LBlock.h"

//int main() {
//    //std::vector<std::map<double, size_t>> u_distributions = compute_u_distributions_grouped(16);
//    //save_u_distributions_to_file(u_distributions, "u_distributions.txt");
//
//    //std::vector<std::vector<double>> precomputed_sums = precompute_partial_sums(u_distributions, 1000000);
//
//    //precomputePhiSystemToFile("Phi_system.txt");
//
//    //std::vector<std::vector<Template>> Phi_system;
//    //loadFullPhiSystem("Phi_system.txt", Phi_system);
//
//    return 0;
//}

int main() {
    constexpr int rounds = 32;
    double p = 0.25; 
    std::string log_filename = "debug.csv";
    std::string phi_filename = "Phi_system.txt";

    std::vector<std::vector<miniTemplate>> Phi_system;
    loadMiniPhiSystem(phi_filename, Phi_system);

    std::ofstream log(log_filename);
    if (!log.is_open()) {
        std::cerr << "Cannot open log file: " << log_filename << "\n";
        return 1;
    }

    log << "rho_perm,p_perm,t,max_value,a_val,b_val,a,b\n";

    auto u_maps = load_u_distributions_from_file("u_distributions.csv");
    auto partial_sums = precompute_partial_sums(u_maps, 100000);

    std::string ub_file = "debug_UB_tmp.csv";

    computeUB(Phi_system, rounds, ub_file, p, log, u_maps, partial_sums);
    log.close();

    std::cout << "UB computation finished.\n";
    return 0;
}