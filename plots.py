import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

# Define constant category order and color mapping
CATEGORY_ORDER = ['Nuts','Dairy and Egg', 'Meats and Fish', 'Legumes', 'Cereal', 'Other']
COLOR_MAPPING = {
    'Dairy and Egg': '#0D5B11',  # Darkest green
    'Meats and Fish': '#187C19',
    'Legumes': '#69B41E',
    'Cereal': '#8DC71E',
    'Nuts': '#224400',          # Light green
    'Other': '#A3D14B'          # Lightest green
}

def create_stacked_horizontal_bar_graph(percentages):
    """
    Create a stacked horizontal bar graph with the given percentages.
    
    Args:
        percentages (dict): Dictionary with categories as keys and percentages as values
    """
    # Values for each category in consistent order
    values = [percentages[cat] for cat in CATEGORY_ORDER]
    colors = [COLOR_MAPPING[cat] for cat in CATEGORY_ORDER]
    
    # Create horizontal bar plot with adjusted dimensions to stay within limits
    # Using 16:1 ratio but smaller absolute size
    fig, ax = plt.subplots(figsize=(16, 1), dpi=600)
    
    # Create stacked bars with increased height
    left = 0
    bars = []
    for i, value in enumerate(values):
        bar = ax.barh(0, value, left=left, color=colors[i], label=CATEGORY_ORDER[i], height=0.6)
        bars.append(bar)
        left += value
    
    # Remove all axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove all spines/borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add percentage labels without borders
    left = 0
    for i, value in enumerate(values):
        center_position = left + value/2
        ax.text(center_position, 0,
                f'{value:.1f}%', 
                va='center',
                ha='center',
                fontsize=12,
                color='white',
                weight='bold')  # Removed path_effects
        left += value
    
    # Adjust legend size
    legend = ax.legend(bbox_to_anchor=(0.5, -0.6), 
                      loc='center', 
                      ncol=2,
                      frameon=False,
                      fontsize=10,  # Reduced from 20
                      columnspacing=2,
                      handlelength=3,
                      handleheight=1.5)
    
    # Adjust height of the bar
    ax.set_ylim(-0.3, 0.3)
    
    # Add more padding around the plot
    plt.tight_layout(pad=2)
    
    # Ensure the figure has a high-quality save
    return fig, ax

# Example usage:
percentages = {
    'Dairy and Egg': 17,
    'Meats and Fish': 36,
    'Legumes': 8,
    'Cereal': 16, 
    'Nuts' : 8, 
    'Other': 14
}

fig, ax = create_stacked_horizontal_bar_graph(percentages)
plt.savefig('output.png', dpi=1200, bbox_inches='tight', format='png')
# or for vector format (recommended for print):
plt.savefig('output.pdf', bbox_inches='tight', format='pdf')