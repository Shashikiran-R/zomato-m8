// Use localhost for local development, and the Railway URL for production
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://localhost:8000' 
  : 'https://web-production-bfbdb.up.railway.app'; // Replace with your actual Railway URL

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const locationSelect = document.getElementById('location-select');
  const cuisineSelect = document.getElementById('cuisine-select');
  const budgetPills = document.querySelectorAll('.budget-pill');
  const ratingSlider = document.getElementById('rating-range');
  const ratingVal = document.getElementById('rating-val');
  const cravingInput = document.getElementById('craving-input');
  const searchBtn = document.getElementById('search-btn');
  const resetBtn = document.getElementById('reset-filters');
  const shimmer = document.getElementById('loading-shimmer');
  const restaurantFeed = document.getElementById('restaurant-feed');
  const summaryHeadline = document.getElementById('summary-headline');
  const summaryText = document.getElementById('summary-text');
  const aiSummaryBanner = document.getElementById('ai-summary-banner');
  const fallbackNote = document.getElementById('fallback-note');
  const errorMessage = document.getElementById('error-message');

  // Interactive micro-behaviors
  // Rating slider dynamic label
  if (ratingSlider && ratingVal) {
    ratingSlider.addEventListener('input', (e) => {
      ratingVal.textContent = `${parseFloat(e.target.value).toFixed(1)}★ and above`;
    });
  }

  // Budget pill single selection
  let selectedBudget = null;
  budgetPills.forEach(pill => {
    pill.addEventListener('click', () => {
      budgetPills.forEach(p => {
        p.setAttribute('data-active', 'false');
        p.classList.remove('bg-secondary', 'text-on-secondary');
        p.classList.add('bg-surface-container', 'text-on-surface-variant');
      });
      pill.setAttribute('data-active', 'true');
      pill.classList.remove('bg-surface-container', 'text-on-surface-variant');
      pill.classList.add('bg-secondary', 'text-on-secondary');
      
      const val = pill.getAttribute('data-value');
      selectedBudget = val === 'Any' ? null : val;
    });
  });

  // Reset button
  if (resetBtn && cravingInput) {
    resetBtn.addEventListener('click', () => {
      cravingInput.value = '';
      if (ratingSlider) {
        ratingSlider.value = 0.0;
        ratingVal.textContent = '0.0★ and above';
      }
      locationSelect.value = 'Any';
      cuisineSelect.value = 'Any';
      // reset budget
      budgetPills.forEach(p => {
        const val = p.getAttribute('data-value');
        if(val === 'Any') {
          p.setAttribute('data-active', 'true');
          p.classList.remove('bg-surface-container', 'text-on-surface-variant');
          p.classList.add('bg-secondary', 'text-on-secondary');
        } else {
          p.setAttribute('data-active', 'false');
          p.classList.remove('bg-secondary', 'text-on-secondary');
          p.classList.add('bg-surface-container', 'text-on-surface-variant');
        }
      });
      selectedBudget = null;
    });
  }

  // Fetch initial metadata
  async function fetchMetadata(endpoint, selectElement) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      
      data.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
      });
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
    }
  }

  fetchMetadata('/locations', locationSelect);
  fetchMetadata('/cuisines', cuisineSelect);

  // Search logic
  searchBtn.addEventListener('click', async () => {
    // Show loading state
    shimmer.classList.remove('hidden');
    restaurantFeed.innerHTML = '';
    errorMessage.classList.add('hidden');
    fallbackNote.classList.add('hidden');
    aiSummaryBanner.classList.add('hidden');
    summaryHeadline.textContent = 'Curating...';
    summaryText.textContent = 'Our AI is analysing restaurants for you.';

    const payload = {
      location: locationSelect.value === 'Any' ? null : locationSelect.value,
      budget: selectedBudget,
      cuisine: cuisineSelect.value === 'Any' ? null : cuisineSelect.value,
      min_rating: parseFloat(ratingSlider.value),
      additional_preferences: cravingInput.value.trim() || null
    };

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.status === 404) {
        showError('No restaurants matched your criteria. Try broadening your filters — for example, pick "Any" for location or cuisine.');
        return;
      }
      
      if (response.status === 422) {
        const errorData = await response.json();
        showError(`Validation error: ${errorData.detail}`);
        return;
      }
      
      if (response.status >= 500) {
        const errorData = await response.json();
        showError(`Server error: ${errorData.detail}`);
        return;
      }

      const data = await response.json();
      renderRecommendations(data);

    } catch (error) {
      console.error('Fetch error:', error);
      showError('Cannot reach the backend API. Make sure the FastAPI server is running at ' + API_BASE_URL);
    } finally {
      shimmer.classList.add('hidden');
    }
  });

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
    summaryHeadline.textContent = 'No results.';
    summaryText.textContent = 'Please try adjusting your filters.';
  }

  function renderRecommendations(data) {
    const { recommendations, summary, fallback_used, fallback_note } = data;

    if (fallback_used && fallback_note) {
      fallbackNote.innerHTML = `⚠️ <strong>Filters relaxed:</strong> ${fallback_note}`;
      fallbackNote.classList.remove('hidden');
    }

    if (summary) {
      aiSummaryBanner.innerHTML = `🤖 <strong>AI Summary:</strong> ${summary}`;
      aiSummaryBanner.classList.remove('hidden');
    }

    if (!recommendations || recommendations.length === 0) {
      showError("No recommendations returned. Try different preferences.");
      return;
    }

    summaryHeadline.textContent = 'Curated for you.';
    summaryText.textContent = `${recommendations.length} handpicked culinary experiences matching your mood & palate.`;

    // Render cards
    recommendations.forEach(rec => {
      const card = document.createElement('article');
      card.className = 'w-full bg-surface-container-lowest rounded-lg p-6 border border-surface-container hover:border-outline-variant transition-all duration-200 shadow-sm flex flex-col sm:flex-row items-start gap-6';

      const cuisinesList = rec.cuisines ? rec.cuisines.split(',').map(c => c.trim()).filter(c => c) : [];
      const cuisinesHtml = cuisinesList.slice(0, 5).map(tag => 
        `<span class="inline-block bg-secondary-fixed text-on-secondary-fixed px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider mr-1 mt-1">${tag}</span>`
      ).join('');

      const deliveryHtml = rec.has_online_delivery 
        ? `<span class="inline-block bg-primary-fixed text-on-primary-fixed px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider mr-1 mt-1">🚚 Delivery</span>`
        : `<span class="inline-block bg-error-container text-on-error-container px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider mr-1 mt-1">No Delivery</span>`;
        
      const bookingHtml = rec.has_table_booking 
        ? `<span class="inline-block bg-primary-fixed text-on-primary-fixed px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider mt-1">📅 Booking</span>`
        : `<span class="inline-block bg-error-container text-on-error-container px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider mt-1">No Booking</span>`;

      // A simple placeholder image assignment based on restaurant name length to keep it somewhat deterministic but varied visually
      const placeholderUrls = [
        "https://lh3.googleusercontent.com/aida-public/AB6AXuACLb44AG_FgWOvcotiUPDjD8mhcNk8eloUQLZsdjngrkhc4VEkvhXu2O1JL0kXwCap2JolXqpUxUeGinC006UiVuwyjEaBxCC5QErnGWBfDBftUgdnxV1pvrKrf21W25Akc52MhaO6BcATACMdC8NL1kEBmZR6n_w2oxARSAeq3cjeGPAgXtZ0c6a90Q8Xp30zVgdEx9c5GJX-7rZLg0aC5wDIuFf78skeUMpBLz8",
        "https://lh3.googleusercontent.com/aida-public/AB6AXuCUMiohs0LgZ8O7NjTXgoBpfnNTTKFeVKU7FGwdCO5WWm1hc82zZ28DYlwIzLy-PXnRKbpWMhmerqsJpHJY6qsT-TEkSWZu77i70VwORQfygfxd4LACfOhuz1bOCGQC6g8zULMAQCxsS-jvf9mePKeuiWRWXK83AXfI4ef7jw4hKrkg1QK3KlNpvvoOBGutASOGLbZylfh_SZKl1iZ1hP7CVqsWZB2lcMQrws6knKM",
        "https://lh3.googleusercontent.com/aida-public/AB6AXuAYcXeIaqlictWyqy-Ru3XWf5hsAL87pdS6WE3M5QjD0JHTm0wfRtVyhZKmMbFnyMXSsZYJ955h10a5Ww9PCHsQ7iaMy6x8jL_8yyvGqNkP1lSeFkU1MYGhi3luMKmixXcd0-Q3AT6bnLmlrUVrVppdoxVicjLc127RuBVjulespM4vSNTLPAOsPcHgi3wzkhPGwAKFEFMNWY8Z5c1vEB-xpY4PNh4Q_HiTcJFFjfc"
      ];
      const imgUrl = placeholderUrls[rec.name.length % placeholderUrls.length];
      const stars = '★'.repeat(Math.round(rec.aggregate_rating));

      card.innerHTML = `
        <div class="w-[100px] h-[100px] rounded-md overflow-hidden flex-shrink-0 bg-surface-container">
          <img class="w-full h-full object-cover" src="${imgUrl}" alt="${rec.name}">
        </div>
        <div class="flex-1 min-w-0 w-full flex flex-col justify-between">
          <div>
            <div class="flex items-start justify-between gap-4">
              <h3 class="font-headline-sm text-headline-sm text-on-surface font-semibold tracking-tight hover:text-primary transition-colors cursor-pointer truncate">
                ${rec.name}
              </h3>
              <button aria-label="Bookmark" class="save-bookmark text-secondary hover:text-primary transition-colors cursor-pointer flex-shrink-0 p-1" title="Save Restaurant">
                <span class="material-symbols-outlined text-[20px]">bookmark_border</span>
              </button>
            </div>
            <p class="font-body-sm text-body-sm text-secondary mt-1">
              📍 ${rec.location} • ₹${rec.average_cost_for_two} for two • <span class="text-primary font-medium">${stars} ${rec.aggregate_rating.toFixed(1)}</span>
            </p>
            <div class="mt-2">${cuisinesHtml}</div>
            <div class="mt-1">${deliveryHtml} ${bookingHtml}</div>
          </div>
          <blockquote class="mt-3.5 border-l-2 border-primary pl-3 italic font-body-sm text-body-sm text-on-surface leading-relaxed bg-surface-container-low py-1.5 pr-3 rounded-r">
            “${rec.explanation}”
          </blockquote>
        </div>
      `;

      restaurantFeed.appendChild(card);
    });

    // Reattach bookmark listeners for new elements
    const bookmarks = document.querySelectorAll('.save-bookmark');
    bookmarks.forEach(btn => {
      btn.addEventListener('click', () => {
        const icon = btn.querySelector('.material-symbols-outlined');
        if (icon.textContent === 'bookmark_border') {
          icon.textContent = 'bookmark';
          icon.style.fontVariationSettings = "'FILL' 1";
          btn.classList.add('text-primary');
          btn.classList.remove('text-secondary');
        } else {
          icon.textContent = 'bookmark_border';
          icon.style.fontVariationSettings = "'FILL' 0";
          btn.classList.remove('text-primary');
          btn.classList.add('text-secondary');
        }
      });
    });
  }
});
