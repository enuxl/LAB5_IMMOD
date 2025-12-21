#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <exception>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ========================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА
// ========================================================

void showCorrelationTable(const MatrixXd& corr_mat,
    const vector<string>& var_names) {
    cout << "\nТаблица корреляций между независимыми переменными:\n";
    cout << setw(18) << " ";
    for (const auto& name : var_names)
        cout << setw(14) << name.substr(0, 12);
    cout << "\n";

    for (int row = 0; row < corr_mat.rows(); row++) {
        cout << setw(18) << var_names[row].substr(0, 12);
        for (int col = 0; col < corr_mat.cols(); col++) {
            cout << setw(14) << fixed << setprecision(4) << corr_mat(row, col);
        }
        cout << "\n";
    }
}

void showDependentCorrelations(const VectorXd& corr_vec,
    const vector<string>& var_names) {
    cout << "\nКорреляции независимых переменных с зависимой:\n";
    for (int i = 0; i < corr_vec.size(); i++) {
        cout << setw(22) << left << var_names[i]
            << ": " << fixed << setprecision(4) << corr_vec[i] << "\n";
    }
}

// ========================================================
// КЛАСС ДЛЯ ЗАГРУЗКИ И ОБРАБОТКИ ДАННЫХ
// ========================================================

class TimeSeriesData {
private:
    struct RegionData {
        string area_name;
        string area_code;
        vector<double> yearly_values;
    };

    vector<RegionData> all_regions;
    vector<string> year_list;
    vector<string> independent_names;
    vector<int> chosen_years;
    int current_year_idx;

public:
    TimeSeriesData() : current_year_idx(-1) {}

    bool openDataFile(const string& file_path) {
        ifstream data_file(file_path);
        if (!data_file.is_open()) {
            cerr << "Проблема: невозможно открыть файл " << file_path << endl;
            return false;
        }

        string data_line;
        int line_counter = 0;

        while (getline(data_file, data_line)) {
            line_counter++;

            if (data_line.empty() || data_line.find_first_not_of(';') == string::npos) {
                continue;
            }

            while (!data_line.empty() && data_line.back() == ';') {
                data_line.pop_back();
            }

            vector<string> elements;
            stringstream ss(data_line);
            string item;

            while (getline(ss, item, ';')) {
                elements.push_back(item);
            }

            if (elements.size() < 3) continue;

            if (line_counter == 1) {
                continue;
            }
            else if (line_counter == 2) {
                for (size_t idx = 2; idx < elements.size(); idx++) {
                    string year_str = elements[idx];
                    size_t dot_pos = year_str.find(" г.");
                    if (dot_pos != string::npos) {
                        year_str = year_str.substr(0, dot_pos);
                    }
                    year_list.push_back(year_str);
                }
            }
            else {
                RegionData rd;
                rd.area_name = elements[0];
                rd.area_code = elements[1];

                for (size_t idx = 2; idx < elements.size(); idx++) {
                    string val_str = elements[idx];

                    val_str.erase(remove(val_str.begin(), val_str.end(), ' '), val_str.end());
                    val_str.erase(remove(val_str.begin(), val_str.end(), '\"'), val_str.end());

                    size_t comma_pos = val_str.find(',');
                    if (comma_pos != string::npos) {
                        val_str[comma_pos] = '.';
                    }

                    string cleaned_value;
                    for (char ch : val_str) {
                        if (ch != ' ') cleaned_value += ch;
                    }

                    try {
                        if (!cleaned_value.empty() && cleaned_value != "-" && cleaned_value != "…" &&
                            cleaned_value != ".." && cleaned_value != "\"\"" && cleaned_value != ".") {
                            rd.yearly_values.push_back(stod(cleaned_value));
                        }
                        else {
                            rd.yearly_values.push_back(numeric_limits<double>::quiet_NaN());
                        }
                    }
                    catch (...) {
                        rd.yearly_values.push_back(numeric_limits<double>::quiet_NaN());
                    }
                }

                int non_nan_count = 0;
                for (double v : rd.yearly_values) {
                    if (!isnan(v)) non_nan_count++;
                }

                if (non_nan_count >= 5) {
                    all_regions.push_back(rd);
                }
            }
        }

        data_file.close();

        if (all_regions.empty()) {
            cerr << "Проблема: данные из файла не загружены." << endl;
            return false;
        }

        cout << "Загружено временных рядов: " << all_regions.size() << endl;
        cout << "Временных отрезков: " << year_list.size() << endl;

        if (!all_regions.empty()) {
            cout << "Пример региона: " << all_regions[0].area_name << endl;
        }

        return true;
    }

    bool prepareAnalysisData(MatrixXd& indep_vars, VectorXd& dep_var, int var_count = 5) {
        vector<vector<double>> indep_rows;
        vector<double> dep_values;

        current_year_idx = year_list.size() - 1;

        independent_names.clear();
        chosen_years.clear();

        for (int i = 0; i < var_count; i++) {
            chosen_years.push_back(current_year_idx - i - 1);
            independent_names.push_back("Yr_" + year_list[current_year_idx - i - 1]);
        }

        for (const auto& region : all_regions) {
            bool valid_entry = true;
            vector<double> row_data;

            for (int year_idx : chosen_years) {
                if (year_idx >= 0 && year_idx < (int)region.yearly_values.size() &&
                    !isnan(region.yearly_values[year_idx])) {
                    row_data.push_back(region.yearly_values[year_idx]);
                }
                else {
                    valid_entry = false;
                    break;
                }
            }

            if (valid_entry && current_year_idx < (int)region.yearly_values.size() &&
                !isnan(region.yearly_values[current_year_idx])) {
                indep_rows.push_back(row_data);
                dep_values.push_back(region.yearly_values[current_year_idx]);
            }
        }

        if (indep_rows.size() < 5) {
            cerr << "Проблема: данных для анализа недостаточно ("
                << indep_rows.size() << " наблюдений)" << endl;
            return false;
        }

        int n_samples = indep_rows.size();
        int n_vars = var_count;

        indep_vars.resize(n_samples, n_vars);
        dep_var.resize(n_samples);

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_vars; j++) {
                indep_vars(i, j) = indep_rows[i][j];
            }
            dep_var(i) = dep_values[i];
        }

        cout << "\nПодготовка данных для анализа:" << endl;
        cout << "  Зависимая переменная: данные " << year_list[current_year_idx] << " года" << endl;
        cout << "  Независимые переменные: " << n_vars << " предыдущих лет" << endl;
        cout << "  Количество наблюдений: " << n_samples << endl;

        return true;
    }

    vector<string> getIndependentNames() const {
        return independent_names;
    }

    int getCurrentYearIndex() const {
        return current_year_idx;
    }

    string getCurrentYearLabel() const {
        if (current_year_idx >= 0 && current_year_idx < (int)year_list.size()) {
            return year_list[current_year_idx];
        }
        return "";
    }

    vector<string> getAreaNames() const {
        vector<string> names;
        for (const auto& region : all_regions) {
            names.push_back(region.area_name);
        }
        return names;
    }

    const RegionData& getAreaInfo(int idx) const {
        return all_regions[idx];
    }
};

// ========================================================
// ФУНКЦИИ ДЛЯ РАСЧЕТА РАСПРЕДЕЛЕНИЙ
// ========================================================

double computeTCumulative(double t_val, int deg_free) {
    double x_val = deg_free / (deg_free + t_val * t_val);
    auto betaContFrac = [](double a_val, double b_val, double x_in) {
        const int ITER_MAX = 100;
        const double PRECISION = 3e-7;
        double qab = a_val + b_val;
        double qap = a_val + 1.0;
        double qam = a_val - 1.0;
        double c_val = 1.0;
        double d_val = 1.0 - qab * x_in / qap;
        if (fabs(d_val) < 1e-30) d_val = 1e-30;
        d_val = 1.0 / d_val;
        double h_val = d_val;
        for (int iter = 1; iter <= ITER_MAX; ++iter) {
            int m2 = 2 * iter;
            double aa = iter * (b_val - iter) * x_in / ((qam + m2) * (a_val + m2));
            d_val = 1.0 + aa * d_val;
            if (fabs(d_val) < 1e-30) d_val = 1e-30;
            c_val = 1.0 + aa / c_val;
            if (fabs(c_val) < 1e-30) c_val = 1e-30;
            d_val = 1.0 / d_val;
            h_val *= d_val * c_val;
            aa = -(a_val + iter) * (qab + iter) * x_in / ((a_val + m2) * (qap + m2));
            d_val = 1.0 + aa * d_val;
            if (fabs(d_val) < 1e-30) d_val = 1e-30;
            c_val = 1.0 + aa / c_val;
            if (fabs(c_val) < 1e-30) c_val = 1e-30;
            d_val = 1.0 / d_val;
            double delta_val = d_val * c_val;
            h_val *= delta_val;
            if (fabs(delta_val - 1.0) < PRECISION) break;
        }
        return h_val;
    };

    double a_val = deg_free / 2.0;
    double b_val = 0.5;
    double beta_term = exp(
        lgamma(a_val + b_val) - lgamma(a_val) - lgamma(b_val)
        + a_val * log(x_val) + b_val * log(1.0 - x_val)
    );

    if (t_val >= 0)
        return 1.0 - 0.5 * beta_term * betaContFrac(a_val, b_val, x_val);
    else
        return 0.5 * beta_term * betaContFrac(a_val, b_val, x_val);
}

// ========================================================
// КЛАСС ЛИНЕЙНОЙ РЕГРЕССИИ
// ========================================================

class LinearModel {
private:
    MatrixXd original_X;
    MatrixXd augmented_X;
    VectorXd original_y;
    VectorXd model_coeffs;
    int sample_size;
    int coeff_count;
    vector<string> variable_labels;

public:
    LinearModel() : sample_size(0), coeff_count(0) {}

    bool buildModel(const MatrixXd& X_matrix, const VectorXd& y_vector,
        const vector<string>& labels) {
        if (X_matrix.rows() != y_vector.size()) return false;
        sample_size = X_matrix.rows();
        int var_count = X_matrix.cols();
        coeff_count = var_count + 1;
        original_X = X_matrix;
        original_y = y_vector;
        variable_labels = labels;

        augmented_X.resize(sample_size, coeff_count);
        augmented_X.col(0) = VectorXd::Ones(sample_size);
        augmented_X.block(0, 1, sample_size, var_count) = X_matrix;

        model_coeffs = (augmented_X.transpose() * augmented_X)
            .inverse()
            * augmented_X.transpose() * y_vector;

        return true;
    }

    VectorXd makePredictions(const MatrixXd& X_matrix) const {
        MatrixXd X_aug(X_matrix.rows(), coeff_count);
        X_aug.col(0) = VectorXd::Ones(X_matrix.rows());
        X_aug.block(0, 1, X_matrix.rows(), X_matrix.cols()) = X_matrix;
        return X_aug * model_coeffs;
    }

    VectorXd getModelCoefficients() const {
        return model_coeffs;
    }

    vector<string> getVariableLabels() const {
        return variable_labels;
    }

    void evaluateModel(double& r_sq,
        double& adj_r_sq,
        double& root_mse,
        double& mean_ape,
        double& mean_ae,
        VectorXd& std_errors,
        VectorXd& t_values,
        VectorXd& p_values) {
        VectorXd predicted = makePredictions(original_X);
        VectorXd residuals = original_y - predicted;

        double ss_error = residuals.squaredNorm();
        double ss_total = (original_y.array() - original_y.mean()).square().sum();
        r_sq = 1.0 - ss_error / ss_total;
        adj_r_sq = 1.0 - (1.0 - r_sq) * (sample_size - 1) / (sample_size - coeff_count);
        root_mse = sqrt(ss_error / sample_size);
        mean_ae = residuals.array().abs().mean();
        mean_ape = 0.0;
        int valid_count = 0;

        for (int i = 0; i < sample_size; i++) {
            if (fabs(original_y[i]) > 1e-12) {
                mean_ape += fabs(residuals[i] / original_y[i]);
                valid_count++;
            }
        }
        if (valid_count > 0) mean_ape = mean_ape / valid_count * 100.0;

        MatrixXd XtX_inverse = (augmented_X.transpose() * augmented_X).inverse();
        double sigma_squared = ss_error / (sample_size - coeff_count);
        std_errors = (sigma_squared * XtX_inverse.diagonal()).array().sqrt();

        t_values.resize(coeff_count);
        p_values.resize(coeff_count);
        int degrees_free = sample_size - coeff_count;

        for (int i = 0; i < coeff_count; i++) {
            t_values[i] = model_coeffs[i] / std_errors[i];
            double t_absolute = fabs(t_values[i]);
            double cumul_prob = computeTCumulative(t_absolute, degrees_free);
            p_values[i] = 2.0 * (1.0 - cumul_prob);
        }
    }

    double computeFValue() {
        VectorXd predicted = makePredictions(original_X);
        VectorXd residuals = original_y - predicted;
        double ss_error = residuals.squaredNorm();
        double ss_regression = (predicted.array() - original_y.mean()).square().sum();
        return (ss_regression / (coeff_count - 1)) / (ss_error / (sample_size - coeff_count));
    }

    MatrixXd computeVarCorrelations() {
        int var_num = original_X.cols();
        MatrixXd corr_matrix = MatrixXd::Zero(var_num, var_num);

        for (int i = 0; i < var_num; i++) {
            for (int j = i; j < var_num; j++) {
                VectorXd xi = original_X.col(i);
                VectorXd xj = original_X.col(j);
                double mean_i = xi.mean();
                double mean_j = xj.mean();
                double numerator = ((xi.array() - mean_i) * (xj.array() - mean_j)).sum();
                double std_i = sqrt((xi.array() - mean_i).square().sum());
                double std_j = sqrt((xj.array() - mean_j).square().sum());

                if (std_i > 0 && std_j > 0) {
                    corr_matrix(i, j) = numerator / (std_i * std_j);
                    corr_matrix(j, i) = corr_matrix(i, j);
                }
            }
        }
        return corr_matrix;
    }

    VectorXd computeDepVarCorrelations() {
        int var_num = original_X.cols();
        VectorXd corr_vector(var_num);
        double mean_y = original_y.mean();
        double std_y = sqrt((original_y.array() - mean_y).square().sum());

        for (int i = 0; i < var_num; i++) {
            VectorXd xi = original_X.col(i);
            double mean_i = xi.mean();
            double std_i = sqrt((xi.array() - mean_i).square().sum());

            if (std_i > 0 && std_y > 0) {
                double numerator = ((xi.array() - mean_i) * (original_y.array() - mean_y)).sum();
                corr_vector[i] = numerator / (std_i * std_y);
            }
            else {
                corr_vector[i] = 0.0;
            }
        }
        return corr_vector;
    }

    vector<int> findSignificantVars(const VectorXd& p_vals,
        double alpha_level) {
        vector<int> selected_vars;
        for (int i = 1; i < p_vals.size(); i++) {
            if (p_vals[i] < alpha_level) {
                selected_vars.push_back(i - 1);
            }
        }
        return selected_vars;
    }

    vector<int> identifyCollinearVars(double corr_limit) {
        MatrixXd corr_matrix = computeVarCorrelations();
        VectorXd corr_with_y = computeDepVarCorrelations();
        vector<int> vars_to_exclude;

        for (int i = 0; i < corr_matrix.rows(); i++) {
            for (int j = i + 1; j < corr_matrix.cols(); j++) {
                if (fabs(corr_matrix(i, j)) > corr_limit) {
                    if (fabs(corr_with_y[i]) < fabs(corr_with_y[j])) {
                        if (find(vars_to_exclude.begin(), vars_to_exclude.end(), i) == vars_to_exclude.end())
                            vars_to_exclude.push_back(i);
                    }
                    else {
                        if (find(vars_to_exclude.begin(), vars_to_exclude.end(), j) == vars_to_exclude.end())
                            vars_to_exclude.push_back(j);
                    }
                }
            }
        }
        return vars_to_exclude;
    }

    LinearModel createReducedModel(const vector<int>& indices) {
        if (indices.empty()) return *this;

        int new_var_count = original_X.cols() - indices.size();
        MatrixXd X_reduced(sample_size, new_var_count);
        vector<string> new_labels;
        int col_idx = 0;

        for (int i = 0; i < original_X.cols(); i++) {
            if (find(indices.begin(), indices.end(), i) == indices.end()) {
                X_reduced.col(col_idx) = original_X.col(i);
                new_labels.push_back(variable_labels[i]);
                col_idx++;
            }
        }

        LinearModel reduced_model;
        reduced_model.buildModel(X_reduced, original_y, new_labels);
        return reduced_model;
    }
};

// ============================================================
// ФУНКЦИЯ ДЛЯ ЗАПИСИ РЕЗУЛЬТАТОВ
// ============================================================

void writeAnalysisResults(const string& output_file,
    const VectorXd& coefficients,
    const vector<string>& labels,
    double r2_val, double adj_r2_val,
    double rmse_val, double mape_val, double mae_val,
    double f_val,
    const VectorXd& p_vals) {
    ofstream out_stream(output_file);
    if (!out_stream.is_open()) {
        cerr << "Проблема с сохранением файла\n";
        return;
    }

    out_stream << fixed << setprecision(6);
    out_stream << "РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА\n";
    out_stream << "=================================\n\n";

    out_stream << "Коэффициенты модели:\n";
    out_stream << "Константа: " << coefficients[0]
        << " (p = " << p_vals[0] << ")\n";

    for (size_t i = 0; i < labels.size(); i++) {
        out_stream << labels[i] << ": "
            << coefficients[i + 1]
            << " (p = " << p_vals[i + 1] << ")\n";
    }

    out_stream << "\nКачество модели:\n";
    out_stream << "R2: " << r2_val << "\n";
    out_stream << "R2 adj: " << adj_r2_val << "\n";
    out_stream << "F-stat: " << f_val << "\n";
    out_stream << "RMSE: " << rmse_val << "\n";
    out_stream << "MAE: " << mae_val << "\n";
    out_stream << "MAPE: " << mape_val << "%\n";

    out_stream.close();
}

// ============================================================
// ОСНОВНАЯ ПРОГРАММА
// ============================================================

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "=============================================\n";
    cout << "АНАЛИЗ ЛИНЕЙНОЙ РЕГРЕССИИ\n";
    cout << "=============================================\n\n";

    try {
        // ВЫБОР ФАЙЛА
        cout << "Укажите файл с данными (по умолчанию DataV8.csv): ";
        string file_name;
        getline(cin, file_name);

        if (file_name.empty()) {
            file_name = "DataV8.csv";
        }

        // ЗАГРУЗКА ДАННЫХ
        cout << "\n1. ЗАГРУЗКА ДАННЫХ\n";
        cout << "------------------\n";

        TimeSeriesData data_handler;
        if (!data_handler.openDataFile(file_name)) {
            cerr << "Ошибка загрузки данных.\n";
            return 1;
        }

        // ПОДГОТОВКА ДАННЫХ
        cout << "\n2. ПОДГОТОВКА ДАННЫХ\n";
        cout << "--------------------\n";

        MatrixXd X_matrix;
        VectorXd y_vector;

        int var_number = 5;
        cout << "Введите число независимых переменных (по умолчанию 5): ";
        string user_input;
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                var_number = stoi(user_input);
                if (var_number < 2) var_number = 2;
                if (var_number > 10) var_number = 10;
            }
            catch (...) {
                var_number = 5;
            }
        }

        if (!data_handler.prepareAnalysisData(X_matrix, y_vector, var_number)) {
            cerr << "Ошибка подготовки данных для анализа.\n";
            return 1;
        }

        // УСТАНОВКА ПАРАМЕТРОВ
        cout << "\n3. УСТАНОВКА ПАРАМЕТРОВ\n";
        cout << "----------------------\n";

        double sig_level = 0.05;
        cout << "Укажите уровень значимости (по умолчанию 0.05): ";
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                sig_level = stod(user_input);
                if (sig_level <= 0) sig_level = 0.05;
                if (sig_level >= 1) sig_level = 0.05;
            }
            catch (...) {
                sig_level = 0.05;
            }
        }

        double colin_threshold = 0.8;
        cout << "Укажите порог мультиколлинеарности (по умолчанию 0.8): ";
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                colin_threshold = stod(user_input);
                if (colin_threshold <= 0) colin_threshold = 0.8;
                if (colin_threshold >= 1) colin_threshold = 0.8;
            }
            catch (...) {
                colin_threshold = 0.8;
            }
        }

        // ПОСТРОЕНИЕ МОДЕЛИ
        cout << "\n4. ПОСТРОЕНИЕ МОДЕЛИ\n";
        cout << "-------------------\n";

        vector<string> var_labels = data_handler.getIndependentNames();
        LinearModel regression_model;

        if (!regression_model.buildModel(X_matrix, y_vector, var_labels)) {
            cerr << "Модель не может быть построена.\n";
            return 1;
        }

        cout << "Модель успешно построена\n";

        // ВЫЧИСЛЕНИЕ СТАТИСТИК
        double r2, adj_r2, rmse, mape, mae;
        VectorXd se, t_stats, p_vals;
        regression_model.evaluateModel(r2, adj_r2, rmse, mape, mae, se, t_stats, p_vals);
        double f_value = regression_model.computeFValue();

        // ВЫВОД РЕЗУЛЬТАТОВ
        cout << "\n5. РЕЗУЛЬТАТЫ АНАЛИЗА\n";
        cout << "--------------------\n";

        cout << "\nКоэффициенты модели:\n";
        VectorXd coeffs = regression_model.getModelCoefficients();
        cout << "Константа: " << coeffs[0]
            << " (p=" << p_vals[0] << ")\n";

        for (size_t i = 0; i < var_labels.size(); i++) {
            cout << var_labels[i] << ": "
                << coeffs[i + 1]
                << " (p=" << p_vals[i + 1] << ")\n";
        }

        cout << "\nR2 = " << r2
            << "\nСкорр. R2 = " << adj_r2
            << "\nF-статистика = " << f_value
            << "\nСКО = " << rmse
            << "\nСр.абс.ошибка = " << mae
            << "\nСр.отн.ошибка = " << mape << "%\n";

        // ПРОВЕРКА КОЛЛИНЕАРНОСТИ
        cout << "\n6. ПРОВЕРКА КОЛЛИНЕАРНОСТИ\n";
        cout << "--------------------------\n";

        MatrixXd corr_matrix = regression_model.computeVarCorrelations();
        showCorrelationTable(corr_matrix, var_labels);

        VectorXd corr_with_y = regression_model.computeDepVarCorrelations();
        showDependentCorrelations(corr_with_y, var_labels);

        vector<int> collinear_vars = regression_model.identifyCollinearVars(colin_threshold);
        if (!collinear_vars.empty()) {
            cout << "\nОбнаружена коллинеарность у переменных:\n";
            for (int idx : collinear_vars) {
                cout << "  - " << var_labels[idx] << endl;
            }

            cout << "\nИсключить проблемные переменные? (y/n): ";
            getline(cin, user_input);

            if (!user_input.empty() && tolower(user_input[0]) == 'y') {
                LinearModel updated_model = regression_model.createReducedModel(collinear_vars);
                regression_model = updated_model;
                var_labels = regression_model.getVariableLabels();

                regression_model.evaluateModel(r2, adj_r2, rmse, mape, mae, se, t_stats, p_vals);
                f_value = regression_model.computeFValue();
                coeffs = regression_model.getModelCoefficients();

                cout << "\nМодель обновлена. Новые коэффициенты:\n";
                cout << "Константа: " << coeffs[0] << " (p=" << p_vals[0] << ")\n";
                for (size_t i = 0; i < var_labels.size(); i++) {
                    cout << var_labels[i] << ": " << coeffs[i + 1] << " (p=" << p_vals[i + 1] << ")\n";
                }
            }
        }

        // ОТБОР ЗНАЧИМЫХ ПЕРЕМЕННЫХ
        cout << "\n7. ОТБОР ЗНАЧИМЫХ ПЕРЕМЕННЫХ\n";
        cout << "---------------------------\n";

        vector<int> significant_vars = regression_model.findSignificantVars(p_vals, sig_level);
        if (!significant_vars.empty()) {
            cout << "Значимые переменные (p < " << sig_level << "):\n";
            for (int idx : significant_vars) {
                cout << "  ok " << var_labels[idx]
                    << " (p = " << scientific << setprecision(2) << p_vals[idx + 1] << ")\n";
            }
        }
        else {
            cout << "Значимых переменных на уровне " << sig_level << " не обнаружено\n";
        }

        // ЗАПИСЬ РЕЗУЛЬТАТОВ
        cout << "\n8. ЗАПИСЬ РЕЗУЛЬТАТОВ\n";
        cout << "-------------------\n";

        writeAnalysisResults("analysis_results_v8.txt",
            coeffs, var_labels,
            r2, adj_r2,
            rmse, mape, mae,
            f_value, p_vals);

        cout << "Анализ успешно завершен\n";
        cout << "Результаты сохранены в: analysis_results_v8.txt\n";

    }
    catch (const exception& e) {
        cerr << "Ошибка выполнения: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
