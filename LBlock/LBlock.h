#pragma once
#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <array>
#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cstdio>  
#include <filesystem>

constexpr int word_size = 4; //u
constexpr int num_blocks = 16; //m
constexpr int bit_size = word_size * num_blocks; //n

using BitVector = std::bitset<bit_size>;
using Template = std::bitset<num_blocks>;
using TemplateID = int64_t;
using miniTemplate = std::bitset<8>;

int hammingWeight(const miniTemplate& t);

miniTemplate P(const miniTemplate& input);
miniTemplate Rho(const miniTemplate& input);
miniTemplate RhoInverse(const miniTemplate& input);

std::set<std::string> computePhiSystem(const Template& a_template, const Template& b_template);
void precomputePhiSystemToFile(const std::string& filename);
void loadMiniPhiSystem(const std::string& filename, std::vector<std::vector<miniTemplate>>& Phi_system);

//UB^t[a][b] upper bound matrix type
using UBMatrix = std::unordered_map<TemplateID, std::unordered_map<TemplateID, double>>;

//void analyzePatternsFromLog(const std::string& log_filename);

std::array<std::array<int, 16>, 16> computeDDT();
std::map<double, size_t> computeU1(const std::array<std::array<int, 16>, 16>& DDT);
std::map<double, size_t> convolveDist(const std::map<double, size_t>& uA, const std::map<double, size_t>& u1);
std::vector<std::map<double, size_t>> compute_u_distributions_grouped(int max_A);
void save_u_distributions_to_file(const std::vector<std::map<double, size_t>>& u_maps, const std::string& filename);
std::vector<std::map<double, size_t>> load_u_distributions_from_file(const std::string& filename);
double sum_u_i_limited(int w, size_t limit, const std::vector<std::map<double, size_t>>& u_distributions);
std::vector<std::vector<double>> precompute_partial_sums(const std::vector<std::map<double, size_t>>& u_distributions, size_t max_limit_per_A);
double get_precomputed_sum(int w, size_t limit, const std::vector<std::vector<double>>& partial_sums);

void computeUB(
    const std::vector<std::vector<miniTemplate>>& Phi_system,
    int rounds,
    const std::string& output_prefix,
    double p,
    std::ofstream& log,
    const std::vector<std::map<double, size_t>>& u_distributions,
    const std::vector<std::vector<double>>& precomputed_sums);