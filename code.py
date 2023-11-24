import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate derivative and plot the functions
def plot_functions(original_function_str):
    # Define the variable
    x = sp.symbols('x')

    try:
        # Parse the user input to a symbolic expression
        original_function = sp.sympify(original_function_str)

        # Calculate the derivative of the function
        derivative_function = sp.diff(original_function, x)

        # Convert the symbolic derivative to a Python function
        derivative_func = sp.lambdify(x, derivative_function, 'numpy')

        # Generate x values for plotting
        x_values = np.linspace(-10, 10, 400)
        y_values_original = [original_function.subs(x, val) for val in x_values]
        y_values_derivative = derivative_func(x_values)

        # Plot the original function and its derivative
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(x_values, y_values_original, label='Original Function')
        plt.title('Original Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x_values, y_values_derivative, label='Derivative')
        plt.title('Derivative of the Function')
        plt.xlabel('x')
        plt.ylabel("f'(x)")
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)  # y-axis
        plt.legend()

        st.pyplot(plt)

    except sp.SympifyError:
        st.error("Invalid function input. Please enter a valid mathematical expression.")

# Streamlit app
def main():
    st.title("Function and Derivative Plotter")

    # User input for the function
    original_function_str = st.text_input("Enter a mathematical function:", "x**2")

    # Button to trigger the plot
    if st.button("Plot"):
        plot_functions(original_function_str)

# Run the Streamlit app
if __name__ == "__main__":
    main()
