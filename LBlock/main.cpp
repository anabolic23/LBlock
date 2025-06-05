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
    std::string output_prefix = "UB";
    std::string log_filename = "UB_log.csv";
    std::ofstream log(log_filename);
    std::string phi_filename = "Phi_system.txt";

    std::vector<std::vector<miniTemplate>> Phi_system;
    loadMiniPhiSystem(phi_filename, Phi_system);

    std::vector<std::map<double, size_t>> u_dist = load_u_distributions_from_file("u_distributions.csv");
    std::vector<std::vector<double>> precomputed_sums = precompute_partial_sums(u_dist, 1000000);

    computeUB_to_bin(Phi_system, rounds, output_prefix, p, log, u_dist, precomputed_sums);

    std::cout << "UB computation finished.\n";
    return 0;
}