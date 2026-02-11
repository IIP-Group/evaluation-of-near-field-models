from src.ris import plot_ris, plot_ris_freq
from src.ula import plot_ula, plot_ula_freq, plot_ula_obstacle

if __name__ == "__main__":
    plot_ula_freq(input_vec_str="pc", used_model_str="pc")
    plot_ula(input_vec_str="pc", used_model_str="pc", region_of_interest="rect")
    plot_ula_obstacle(input_vec_str="pc", used_model_str="pc", region_of_interest="rect")
    plot_ris(input_coefficent_str="pc", used_model_str="pc", region_of_interest="rect")
    plot_ris_freq(input_coefficent_str="pc", used_model_str="pc")
