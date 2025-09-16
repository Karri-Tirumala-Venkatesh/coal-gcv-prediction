// Application data and configuration
const APP_DATA = {
    coalSampleSources: ["Central India Coal", "Southern India Coal"],
    analysisEnvironments: ["On Air Dried Basis", "Moist on 60% RH & 40°C"],
    gcvRanges: {
        low: { min: 0, max: 3500, label: "Low Grade Coal", cssClass: "status--low-grade" },
        medium: { min: 3500, max: 4500, label: "Medium Grade Coal", cssClass: "status--medium-grade" },
        good: { min: 4500, max: 5500, label: "Good Grade Coal", cssClass: "status--good-grade" },
        high: { min: 5500, max: 10000, label: "High Grade Coal", cssClass: "status--high-grade" }
    },
    predictionModel: {
        baseConstant: 7115.197,
        moistureCoefficient: -123.971,
        ashCoefficient: -81.3121,
        fixedCarbonCoefficient: 20.7421,
        volatileMatterFactor: 18.5,
        regionalAdjustments: {
            "Central India Coal": 0,
            "Southern India Coal": -150
        },
        environmentAdjustments: {
            "On Air Dried Basis": 0,
            "Moist on 60% RH & 40°C": -200
        }
    }
};

// Only handles form, validation, and displaying results

let formElements = {};
let resultElements = {};

document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    bindEventListeners();
    updateCalculatedFixedCarbon();
});

function initializeElements() {
    formElements = {
        form: document.getElementById('gcvForm'),
        coalSource: document.getElementById('coalSource'),
        analysisEnvironment: document.getElementById('analysisEnvironment'),
        moisture: document.getElementById('moisture'),
        ash: document.getElementById('ash'),
        volatileMatter: document.getElementById('volatileMatter'),
        fixedCarbon: document.getElementById('fixedCarbon'),
        predictBtn: document.getElementById('predictBtn'),
        clearBtn: document.getElementById('clearBtn'),
        calculatedFC: document.getElementById('calculatedFC')
    };
    resultElements = {
        resultsSection: document.getElementById('resultsSection'),
        gcvValue: document.getElementById('gcvValue'),
        calculationDetails: document.getElementById('calculationDetails')
    };
}

function bindEventListeners() {
    formElements.form.addEventListener('submit', handleFormSubmit);
    formElements.clearBtn.addEventListener('click', clearForm);
    [formElements.moisture, formElements.ash, formElements.volatileMatter].forEach(input => {
        input.addEventListener('input', updateCalculatedFixedCarbon);
    });
    Object.values(formElements).forEach(element => {
        if (element && (element.tagName === 'INPUT' || element.tagName === 'SELECT')) {
            element.addEventListener('blur', validateField);
            element.addEventListener('input', clearFieldError);
        }
    });
}

function handleFormSubmit(event) {
    event.preventDefault();
    if (!validateForm()) return;
    showLoading(true);

    const data = {
        coalSource: formElements.coalSource.value,
        analysisEnvironment: formElements.analysisEnvironment.value,
        moisture: formElements.moisture.value,
        ash: formElements.ash.value,
        volatileMatter: formElements.volatileMatter.value,
        fixedCarbon: formElements.fixedCarbon.value || updateCalculatedFixedCarbon()
    };

    fetch('/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        displayResults(result);
        showLoading(false);
    })
    .catch(() => {
        alert('Prediction failed. Please try again.');
        showLoading(false);
    });
}

function getCSRFToken() {
    const name = 'csrftoken';
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        let [key, value] = cookie.trim().split('=');
        if (key === name) return decodeURIComponent(value);
    }
    return '';
}

function validateForm() {
    let isValid = true;
    const requiredFields = ['coalSource', 'analysisEnvironment', 'moisture', 'ash', 'volatileMatter'];
    requiredFields.forEach(fieldName => {
        const field = formElements[fieldName];
        if (!validateField({ target: field })) isValid = false;
    });
    return isValid;
}

function validateField(event) {
    const field = event.target;
    const fieldName = field.name;
    const value = field.value.trim();
    const errorElement = document.getElementById(fieldName + 'Error');
    let errorMessage = '';
    if (field.required && !value) errorMessage = 'This field is required';
    if (errorMessage) {
        errorElement.textContent = errorMessage;
        field.parentElement.classList.add('has-error');
        return false;
    } else {
        errorElement.textContent = '';
        field.parentElement.classList.remove('has-error');
        field.parentElement.classList.add('has-success');
        return true;
    }
}

function clearFieldError(event) {
    const field = event.target;
    const errorElement = document.getElementById(field.name + 'Error');
    if (errorElement) errorElement.textContent = '';
    field.parentElement.classList.remove('has-error');
}

function updateCalculatedFixedCarbon() {
    const moisture = parseFloat(formElements.moisture.value) || 0;
    const ash = parseFloat(formElements.ash.value) || 0;
    const volatileMatter = parseFloat(formElements.volatileMatter.value) || 0;
    const calculatedFC = Math.max(0, 100 - moisture - ash - volatileMatter);
    formElements.calculatedFC.textContent = calculatedFC.toFixed(2);
    return calculatedFC;
}

function displayResults(result) {
    resultElements.resultsSection.classList.remove('hidden');
    resultElements.gcvValue.textContent = result.gcv ? `${result.gcv} kcal/kg` : '--';
    displayCalculationDetails(result);
    resultElements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayCalculationDetails(result) {
    if (!result.inputs) return;
    const details = [
        { label: 'Coal Source', value: result.inputs.coalSource },
        { label: 'Analysis Environment', value: result.inputs.analysisEnvironment },
        { label: 'Moisture (%)', value: parseFloat(result.inputs.moisture).toFixed(2) },
        { label: 'Ash (%)', value: parseFloat(result.inputs.ash).toFixed(2) },
        { label: 'Volatile Matter (%)', value: parseFloat(result.inputs.volatileMatter).toFixed(2) },
        { label: 'Fixed Carbon (%)', value: parseFloat(result.inputs.fixedCarbon).toFixed(2) }
    ];
    resultElements.calculationDetails.innerHTML = details.map(detail => `
        <div class="detail-item">
            <span class="detail-label">${detail.label}:</span>
            <span class="detail-value">${detail.value}</span>
        </div>
    `).join('');
}

function clearForm() {
    formElements.form.reset();
    document.querySelectorAll('.error-message').forEach(element => { element.textContent = ''; });
    document.querySelectorAll('.form-group').forEach(group => { group.classList.remove('has-error', 'has-success'); });
    resultElements.resultsSection.classList.add('hidden');
    formElements.calculatedFC.textContent = '--';
    formElements.coalSource.focus();
}

function showLoading(show) {
    const btnText = formElements.predictBtn.querySelector('.btn__text');
    const btnLoading = formElements.predictBtn.querySelector('.btn__loading');
    if (show) {
        formElements.predictBtn.classList.add('btn--loading');
        formElements.predictBtn.disabled = true;
        btnText.classList.add('hidden');
        btnLoading.classList.remove('hidden');
    } else {
        formElements.predictBtn.classList.remove('btn--loading');
        formElements.predictBtn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
    }
}