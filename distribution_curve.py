import matplotlib.pyplot as plt

# Define the data
categories = ["(-∞,1.5]", "(1.5,2.5]", "(2.5,3.5]", "(3.5,4.5]", "(4.5,5.5]", "(5.5,6.5]", "(6.5,+∞)"]
probabilities = [0.1, 2.4, 15.2, 36.5, 33.1, 11.2, 1.5]

# Create the bar chart
plt.bar(categories, probabilities, color='b')

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Probability (%)')
plt.title('Probability Distribution')

# Display the plot
plt.show()
