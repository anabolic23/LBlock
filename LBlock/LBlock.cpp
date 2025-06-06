﻿#include "LBlock.h"

namespace std {
    template<>
    struct hash<miniTemplate> {
        std::size_t operator()(const miniTemplate& m) const {
            return std::hash<unsigned long long>()(m.to_ullong());
        }
    };
}

int hammingWeight(const miniTemplate& t) {
    return static_cast<int>(t.count());
}

miniTemplate Rho(const miniTemplate& input) {
    miniTemplate result;
    for (int i = 0; i < 8; ++i)
        result[i] = input[(i + 1) % 8];     
    return result;
}

miniTemplate RhoInverse(const miniTemplate& input) {
    miniTemplate result;
    for (int i = 0; i < 8; ++i)
        result[(i + 1) % 8] = input[i]; 
    return result;
}

miniTemplate P(const miniTemplate& input) {
    miniTemplate result;
    result[7] = input[6]; // U7 = Z6
    result[6] = input[4]; // U6 = Z4
    result[5] = input[7]; // U5 = Z7
    result[4] = input[5]; // U4 = Z5
    result[3] = input[2]; // U3 = Z2
    result[2] = input[0]; // U2 = Z0
    result[1] = input[3]; // U1 = Z3
    result[0] = input[1]; // U0 = Z1
    return result;
}

std::set<std::string> computePhiSystem(const Template& a_template, const Template& b_template) {
    std::vector<int> uncertain_indices;
    std::bitset<num_blocks> base_result;

    for (int i = 0; i < num_blocks; ++i) {
        bool ai = a_template[i];
        bool bi = b_template[i];

        if (!ai && !bi) {
            base_result[i] = 0;
        }
        else if ((ai == 1 && bi == 0) || (ai == 0 && bi == 1)) {
            base_result[i] = 1;
        }
        else if (ai == 1 && bi == 1) {
            uncertain_indices.push_back(i);
        }
    }

    std::set<std::string> result;
    int total_variants = 1 << uncertain_indices.size();

    for (int mask = 0; mask < total_variants; ++mask) {
        std::bitset<num_blocks> c = base_result;
        for (size_t j = 0; j < uncertain_indices.size(); ++j) {
            int index = uncertain_indices[j];
            c[index] = (mask >> j) & 1;
        }
        result.insert(c.to_string());
    }

    return result;
}

void precomputePhiSystemToFile(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << " Failed to open file: " << filename << "\n";
        return;
    }


    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
        Template a(a_val);

        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
            Template b(b_val);

            std::set<std::string> phi_set = computePhiSystem(a, b);
            TemplateID line_id = (a_val << 8) + b_val; // a * 256 + b

            //file << line_id << ":";

            for (const auto& phi_str : phi_set) {
                Template phi(phi_str);
                TemplateID phi_val = phi.to_ulong();
                file << " " << phi_val;
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Phi system written to " << filename << "\n";
}

void loadMiniPhiSystem(const std::string& filename, std::vector<std::unordered_set<miniTemplate>>& Phi_system) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open Phi_system.txt\n";
        return;
    }

    constexpr int phi_size = 1 << 16;
    Phi_system.clear();
    Phi_system.resize(phi_size);

    std::string line;
    int line_id = 0;

    while (std::getline(file, line) && line_id < phi_size) {
        std::istringstream iss(line);
        TemplateID phi_val;
        while (iss >> phi_val) {
            Phi_system[line_id].insert(miniTemplate(phi_val));
        }
        ++line_id;
    }

    if (line_id != phi_size) {
        std::cerr << "Warning: expected 256 lines but got " << line_id << "\n";
    }

    file.close();
    std::cout << "Mini Phi system loaded into memory (" << line_id << " entries).\n";
}

const uint8_t LBlock_SBOX[16] = {
    14, 9, 15, 0,
    13, 4, 10, 11,
    1, 2, 8, 3,
    7, 6, 12, 5
};

std::map<double, size_t> u_dist;
std::vector<std::vector<double>> partial_sums;

std::array<std::array<int, 16>, 16> computeDDT() {
    std::array<std::array<int, 16>, 16> DDT = {};

    for (int alpha = 0; alpha < 16; ++alpha) {
        for (int x = 0; x < 16; ++x) {
            int y = LBlock_SBOX[x ^ alpha] ^ LBlock_SBOX[x];
            DDT[alpha][y]++;
        }
    }

    return DDT;
}

std::map<double, size_t> computeU1(const std::array<std::array<int, 16>, 16>& DDT) {
    std::map<double, size_t> u1;

    for (int alpha = 1; alpha < 16; ++alpha) { 
        for (int beta = 0; beta < 16; ++beta) {
            int count = DDT[alpha][beta];
            if (count > 0) {
                double prob = static_cast<double>(count) / 16.0; // 4-bit S-box
                u1[prob]++;
            }
        }
    }

    return u1;
}

std::map<double, size_t> convolveDist(const std::map<double, size_t>& uA, const std::map<double, size_t>& u1) {
    std::map<double, size_t> result;

    for (const std::pair<const double, size_t>& entry1 : uA) {
        double p1 = entry1.first;
        size_t n1 = entry1.second;

        for (const std::pair<const double, size_t>& entry2 : u1) {
            double p2 = entry2.first;
            size_t n2 = entry2.second;

            double prod = p1 * p2;
            result[prod] += n1 * n2;
        }
    }

    return result;
}

std::vector<std::map<double, size_t>> compute_u_distributions_grouped(int max_A) {
    auto DDT = computeDDT();
    auto u1_map = computeU1(DDT);

    std::vector<std::map<double, size_t>> u_maps;
    u_maps.push_back(u1_map); // u1

    for (int A = 2; A <= max_A; ++A) {
        auto uA = convolveDist(u_maps[A - 2], u1_map);
        u_maps.push_back(uA);
        std::cout << "Computed grouped u[" << A << "] (unique probabilities = " << uA.size() << ")\n";
    }


    return u_maps;
}

void save_u_distributions_to_file(const std::vector<std::map<double, size_t>>& u_maps, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    for (size_t A = 0; A < u_maps.size(); ++A) {
        out << "A = " << A + 1 << "\n";
        for (const auto& entry : u_maps[A]) {
            out << std::setprecision(16) << entry.first << "," << entry.second << "\n";
        }
        out << "\n";
    }

    out.close();
    std::cout << "u_i(A) distributions saved to: " << filename << "\n";
}

std::vector<std::map<double, size_t>> load_u_distributions_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::map<double, size_t>> u_maps;
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return u_maps;
    }

    std::string line;
    std::map<double, size_t> current_map;

    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!current_map.empty()) {
                u_maps.push_back(current_map);
                current_map.clear();
            }
            continue;
        }

        if (line.find("A = ") == 0)
            continue;

        std::istringstream iss(line);
        double prob;
        size_t count;
        char comma;
        if (iss >> prob >> comma >> count) {
            current_map[prob] = count;
        }
    }

    if (!current_map.empty()) {
        u_maps.push_back(current_map);
    }

    file.close();
    std::cout << "Loaded u_i(A) distributions from: " << filename << "\n";
    return u_maps;
}

double sum_u_i_limited(int w, size_t limit, const std::vector<std::map<double, size_t>>& u_distributions) {
    if (w <= 0 || w > static_cast<int>(u_distributions.size())) return 0.0;

    const auto& u = u_distributions[w - 1]; // A = w
    double sum = 0.0;
    size_t count = 0;

    for (const auto& entry : u) {
        double prob = entry.first;
        size_t freq = entry.second;

        for (size_t i = 0; i < freq && count < limit; ++i, ++count) {
            sum += prob;
        }

        if (count >= limit)
            break;
    }

    return sum;
}

std::vector<std::vector<double>> precompute_partial_sums(
    const std::vector<std::map<double, size_t>>& u_distributions,
    size_t max_limit_per_A = 1000000
) {
    std::vector<std::vector<double>> partial_sums;

    for (const auto& u : u_distributions) {
        std::vector<double> sums;
        sums.reserve(max_limit_per_A);
        size_t count = 0;
        double current_sum = 0.0;

        for (const auto& entry : u) {
            double prob = entry.first;
            size_t freq = entry.second;

            for (size_t i = 0; i < freq && count < max_limit_per_A; ++i, ++count) {
                current_sum += prob;
                sums.push_back(current_sum);
            }

            if (count >= max_limit_per_A)
                break;
        }

        partial_sums.push_back(sums); // sums[i] = sum_{j=0}^{i} u_j
    }

    return partial_sums;
}

double get_precomputed_sum(
    int w, size_t limit,
    const std::vector<std::vector<double>>& partial_sums
) {
    if (w <= 0 || w > static_cast<int>(partial_sums.size())) return 0.0;
    const auto& vec = partial_sums[w - 1];
    if (limit == 0) return 0.0;
    if (limit >= vec.size()) return vec.back();
    return vec[limit - 1];
}

//void computeUB(
//    const std::vector<std::unordered_set<miniTemplate>>& Phi_system,
//    int rounds,
//    const std::string& output_file,
//    double p,
//    std::ofstream& log,
//    const std::vector<std::map<double, size_t>>& u_distributions,
//    const std::vector<std::vector<double>>& precomputed_sums) {
//    using UBMatrix = std::unordered_map<TemplateID, std::unordered_map<TemplateID, double>>;
//
//    auto to_string = [](const std::vector<int>& perm) -> std::string {
//        std::string s;
//        for (int i : perm) s += std::to_string(i);
//        return s;
//    };
//
//    // Step 1: compute UB^[2]
//    UBMatrix UB_current;
//
//    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
//        Template a(a_val);
//        miniTemplate a_x, a_y;
//        for (int i = 0; i < num_blocks / 2; ++i) a_x[i] = a[i];
//        for (int i = 0; i < num_blocks / 2; ++i) a_y[i] = a[i + num_blocks / 2];
//
//        miniTemplate Pa_y = P(a_y);
//        miniTemplate rho_ax = Rho(a_x);
//        miniTemplate rho_ay = Rho(a_y);
//
//        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
//            Template b(b_val);
//            miniTemplate b_x, b_y;
//            for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
//            for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];
//
//            miniTemplate Pb_x = P(b_x);
//
//            TemplateID line1 = (Pa_y.to_ullong() << 4) + rho_ax.to_ullong();
//            TemplateID line2 = (b_x.to_ullong() << 4) + rho_ay.to_ullong();
//
//            bool cond1 = false, cond2 = false;
//
//            if (line1 < Phi_system.size()) {
//                for (const auto& candidate : Phi_system[line1]) {
//                    if (candidate == b_x) {
//                        cond1 = true;
//                        break;
//                    }
//                }
//            }
//
//            if (line2 < Phi_system.size()) {
//                for (const auto& candidate : Phi_system[line2]) {
//                    if (candidate == b_y) {
//                        cond2 = true;
//                        break;
//                    }
//                }
//            }
//
//            if (cond1 && cond2) {
//                uint8_t wt_a_y = hammingWeight(a_y);
//                uint8_t wt_b_x = hammingWeight(b_x);
//                double result = std::pow(p, wt_a_y + wt_b_x);
//
//                UB_current[a_val][b_val] = result;
//            }
//            else {
//                UB_current[a_val][b_val] = 0.0;
//            }
//        }
//    }
//
//    double max_t2 = 0.0;
//    TemplateID a_max2 = 0, b_max2 = 0;
//
//    for (TemplateID a_val = 1; a_val < (1 << num_blocks); ++a_val) {
//        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
//            double current = UB_current[a_val][b_val];
//            if (current > max_t2) {
//                max_t2 = current;
//                a_max2 = a_val;
//                b_max2 = b_val;
//            }
//        }
//    }
//
//    log << "rho_perm_LBlock,p_perm_LBlock,2,"
//        << std::setprecision(16) << max_t2 << ","
//        << a_max2 << "," << b_max2 << ","
//        << Template(a_max2) << "," << Template(b_max2) << "\n";
//
//    // Step 2: for t = 3, ..., rounds
//    for (int t = 3; t <= rounds; ++t) {
//        UBMatrix UB_next;
//
//        for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
//            for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
//                Template b(b_val);
//                miniTemplate b_x, b_y;
//                for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
//                for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];
//
//                if (b_x.none()) {
//                    miniTemplate rho_inv_by = RhoInverse(b_y);
//
//                    Template transformed_b;
//                    for (int i = 0; i < num_blocks / 2; ++i) {
//                        transformed_b[i] = rho_inv_by[i];
//                        transformed_b[i + num_blocks / 2] = 0;
//                    }
//
//                    TemplateID transformed_b_val = transformed_b.to_ulong();
//                    UB_next[a_val][b_val] = UB_current[a_val][transformed_b_val];
//                    continue;
//                }
//
//                else {
//                    // Case 2.2: b_x != 0
//                    // 2.2.1 find M = max_gamma UB_current(gamma, b)
//                    double M = 0.0;
//                    for (TemplateID gamma_val = 0; gamma_val < (1 << num_blocks); ++gamma_val) {
//                        M = std::max(M, UB_current[gamma_val][b_val]);
//                    }
//
//                    // 2.2.2 compute second candidate
//                    miniTemplate Pb_x = P(b_x);
//                    TemplateID phi_index = (b_y.to_ulong() << (num_blocks / 2)) + Pb_x.to_ulong();
//
//                    int bx_weight = hammingWeight(b_x);
//                    double sum = 0.0;
//                    if (phi_index < Phi_system.size()) {
//                        const auto& phi_set = Phi_system[phi_index];
//
//                        for (TemplateID gamma_val_raw = 0; gamma_val_raw < (1 << (num_blocks / 2)); ++gamma_val_raw) {
//                            miniTemplate gamma;
//                            for (int i = 0; i < num_blocks / 2; ++i)
//                                gamma[i] = (gamma_val_raw >> i) & 1;
//
//                            miniTemplate rho_gamma = Rho(gamma);
//                            if (std::find(phi_set.begin(), phi_set.end(), rho_gamma) != phi_set.end()) {
//                                Template gamma_template;
//                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i] = gamma[i];
//                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i + num_blocks / 2] = b_x[i];
//                                TemplateID gamma_template_val = gamma_template.to_ulong();
//
//                                int gamma_weight = static_cast<int>(gamma.count());
//                                size_t limit = static_cast<size_t>(std::pow((1 << word_size) - 1, gamma_weight));
//
//                                double sum_u = get_precomputed_sum(bx_weight, limit, precomputed_sums);
//
//                                if (UB_current[a_val].find(gamma_template_val) != UB_current[a_val].end()) {
//                                    sum += UB_current[a_val][gamma_template_val] * sum_u;
//                                    //std::cout << " sum = " << std::setprecision(10) << sum;
//                                }
//                            }
//                        }
//                    }
//                    UB_next[a_val][b_val] = std::min(M, sum);
//
//                }
//            }
//        }
//
//        UB_current = UB_next;
//        //std::cout << "Completed round t = " << t << "\n";
//
//        double max_t = 0.0;
//        TemplateID a_max = 0, b_max = 0;
//
//        for (TemplateID a_val = 1; a_val < (1 << num_blocks); ++a_val) {
//            for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
//                double current = UB_current[a_val][b_val];
//                if (current > max_t) {
//                    max_t = current;
//                    a_max = a_val;
//                    b_max = b_val;
//                }
//            }
//        }
//
//        log << "rho_perm_LBlock,p_perm_LBlock," << t << ","
//            << std::setprecision(16) << max_t << ","
//            << a_max << "," << b_max << ","
//            << Template(a_max) << "," << Template(b_max) << "\n";
//
//
//    }
//
//    // Step 3: Write final UB to file
//    std::ofstream file(output_file);
//    if (!file.is_open()) {
//        std::cerr << "Cannot open output file: " << output_file << "\n";
//        return;
//    }
//
//    file << std::fixed << std::setprecision(16);
//
//    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
//        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
//            file << UB_current[a_val][b_val];
//            if (b_val != (1 << num_blocks) - 1) file << ",";
//        }
//        file << "\n";
//    }
//
//    file.close();
//    std::cout << "Final UB matrix written to " << output_file << "\n";
//}

void computeUB(
    const std::vector<std::unordered_set<miniTemplate>>& Phi_system,
    int rounds,
    const std::string& output_file,
    double p,
    std::ofstream& log,
    const std::vector<std::map<double, size_t>>& u_distributions,
    const std::vector<std::vector<double>>& precomputed_sums) {
    using UBMatrix = std::unordered_map<TemplateID, std::unordered_map<TemplateID, double>>;

    // Step 1: compute UB^[2]
    UBMatrix UB_current;

    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
        Template a(a_val);
        miniTemplate a_x, a_y;
        for (int i = 0; i < num_blocks / 2; ++i) a_x[i] = a[i];
        for (int i = 0; i < num_blocks / 2; ++i) a_y[i] = a[i + num_blocks / 2];

        miniTemplate Pa_y = P(a_y);
        miniTemplate rho_ax = Rho(a_x);
        miniTemplate rho_ay = Rho(a_y);

        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
            Template b(b_val);
            miniTemplate b_x, b_y;
            for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
            for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];

            miniTemplate Pb_x = P(b_x);

            TemplateID line1 = (Pa_y.to_ullong() << 8) + rho_ax.to_ullong();
            TemplateID line2 = (b_x.to_ullong() << 8) + rho_ay.to_ullong();

            bool cond1 = false, cond2 = false;

            if (line1 < Phi_system.size()) {
                for (const auto& candidate : Phi_system[line1]) {
                    if (candidate == b_x) {
                        cond1 = true;
                        break;
                    }
                }
            }

            if (line2 < Phi_system.size()) {
                for (const auto& candidate : Phi_system[line2]) {
                    if (candidate == b_y) {
                        cond2 = true;
                        break;
                    }
                }
            }

            if (cond1 && cond2) {
                uint8_t wt_a_y = hammingWeight(a_y);
                uint8_t wt_b_x = hammingWeight(b_x);
                /*std::cout << "Hamming weights: wt(a_y) = " << static_cast<int>(wt_a_y)
                    << ", wt(b_x) = " << static_cast<int>(wt_b_x) << "\n";*/
                double result = std::pow(p, wt_a_y + wt_b_x);
                /*std::cout << "UB[" << a << "][" << b << "] = " << result << "\n";*/

                UB_current[a_val][b_val] = result;
            }
            else {
                UB_current[a_val][b_val] = 0.0;
            }
        }
    }

    double max_t2 = 0.0;
    TemplateID a_max2 = 0, b_max2 = 0;

    for (TemplateID a_val = 1; a_val < (1 << num_blocks); ++a_val) {
        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
            double current = UB_current[a_val][b_val];
            if (current > max_t2) {
                max_t2 = current;
                a_max2 = a_val;
                b_max2 = b_val;
            }
        }
    }

    log << "rho_perm_LBlock,p_perm_LBlock,2,"
        << std::setprecision(16) << max_t2 << ","
        << a_max2 << "," << b_max2 << ","
        << Template(a_max2) << "," << Template(b_max2) << "\n";

    // Step 2: for t = 3, ..., rounds
    for (int t = 3; t <= rounds; ++t) {
        UBMatrix UB_next;

        for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
            for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
                Template b(b_val);
                miniTemplate b_x, b_y;
                for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
                for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];

                if (b_x.none()) {
                    miniTemplate rho_inv_by = RhoInverse(b_y);

                    Template transformed_b;
                    for (int i = 0; i < num_blocks / 2; ++i) {
                        transformed_b[i] = rho_inv_by[i];
                        transformed_b[i + num_blocks / 2] = 0;
                    }

                    TemplateID transformed_b_val = transformed_b.to_ulong();
                    UB_next[a_val][b_val] = UB_current[a_val][transformed_b_val];
                    continue;
                }

                else {
                    // Case 2.2: b_x != 0
                    // 2.2.1 find M = max_gamma UB_current(gamma, b)
                    double M = 0.0;
                    for (TemplateID gamma_val = 0; gamma_val < (1 << num_blocks); ++gamma_val) {
                        M = std::max(M, UB_current[gamma_val][b_val]);
                    }

                    // 2.2.2 compute second candidate
                    miniTemplate Pb_x = P(b_x);
                    TemplateID phi_index = (b_y.to_ulong() << (num_blocks / 2)) + Pb_x.to_ulong();

                    int bx_weight = hammingWeight(b_x);
                    double sum = 0.0;
                    if (phi_index < Phi_system.size()) {
                        const auto& phi_set = Phi_system[phi_index];

                        for (TemplateID gamma_val_raw = 0; gamma_val_raw < (1 << (num_blocks / 2)); ++gamma_val_raw) {
                            miniTemplate gamma;
                            for (int i = 0; i < num_blocks / 2; ++i)
                                gamma[i] = (gamma_val_raw >> i) & 1;

                            miniTemplate rho_gamma = Rho(gamma);
                            if (std::find(phi_set.begin(), phi_set.end(), rho_gamma) != phi_set.end()) {
                                Template gamma_template;
                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i] = gamma[i];
                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i + num_blocks / 2] = b_x[i];
                                TemplateID gamma_template_val = gamma_template.to_ulong();

                                int gamma_weight = static_cast<int>(gamma.count());
                                size_t limit = static_cast<size_t>(std::pow((1 << word_size) - 1, gamma_weight));

                                double sum_u = get_precomputed_sum(bx_weight, limit, precomputed_sums);

                                if (UB_current[a_val].find(gamma_template_val) != UB_current[a_val].end()) {
                                    sum += UB_current[a_val][gamma_template_val] * sum_u;
                                    //std::cout << " sum = " << std::setprecision(10) << sum;
                                }
                            }
                        }
                    }
                    UB_next[a_val][b_val] = std::min(M, sum);
                }
            }
        }

        UB_current = UB_next;
        //std::cout << "Completed round t = " << t << "\n";

        double max_t = 0.0;
        TemplateID a_max = 0, b_max = 0;

        for (TemplateID a_val = 1; a_val < (1 << num_blocks); ++a_val) {
            for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
                double current = UB_current[a_val][b_val];
                if (current > max_t) {
                    max_t = current;
                    a_max = a_val;
                    b_max = b_val;
                }
            }
        }

        log << "rho_perm_LBlock,p_perm_LBlock," << t << ","
            << std::setprecision(16) << max_t << ","
            << a_max << "," << b_max << ","
            << Template(a_max) << "," << Template(b_max) << "\n";

    }

    // Step 3: Write final UB to file
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Cannot open output file: " << output_file << "\n";
        return;
    }

    file << std::fixed << std::setprecision(16);

    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
            file << UB_current[a_val][b_val];
            if (b_val != (1 << num_blocks) - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Final UB matrix written to " << output_file << "\n";
}