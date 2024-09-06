let score = document.querySelector(".score");
let headings = document.querySelectorAll(".Heading");

function getValue() {
    let inputValue = parseInt(document.getElementById('id').value);
    if (isNaN(inputValue)) {
        alert("Please enter a valid number.");
        return;
    }
    let needle = document.querySelector(".needle");
    needle.style.setProperty("--score", inputValue);
    score.innerText = inputValue;
    
    rotateNeedle(inputValue);
}

function rotateNeedle(score) {
    if (score >= 250 && score <= 579) {
        rotateTo(-78);
    } else if (score >= 580 && score <= 669) {
        rotateTo(-40);
    } else if (score >= 670 && score <= 739) {
        rotateTo(0);
    } else if (score >= 740 && score <= 799) {
        rotateTo(40);
    } else if (score >= 800 && score <= 900) {
        rotateTo(78);
    } else {
        alert("Invalid score: Score must be between 250 and 900.");
    }
}

function rotateTo(rotation) {
    document.documentElement.style.setProperty('--rotate', `${rotation}deg`);
    headings.forEach(heading => {
        heading.style.setProperty('--rotate', `${rotation}deg`);
    });
}
