// Simple hover and drag effect for cards and how-step
// Only hover animation interaction for cards and how-step elements
function addCardHoverEffect(selector) {
    const cards = document.querySelectorAll(selector);
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transition = 'box-shadow 0.3s, transform 0.3s';
            card.style.boxShadow = '0 8px 32px rgba(60, 90, 200, 0.18)';
            card.style.transform = 'translateY(-4px) scale(1.03)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transition = 'box-shadow 0.3s, transform 0.3s';
            card.style.boxShadow = '0 4px 24px rgba(60, 90, 200, 0.08)';
            card.style.transform = 'none';
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    addCardHoverEffect('.card');
    addCardHoverEffect('.how-step');
});
