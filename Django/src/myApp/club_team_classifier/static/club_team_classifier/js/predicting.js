function onPredictButtonClick() {
    target = document.getElementById("predicting");
    // target.insertAdjacentHTML('afterbegin', 'Now predicting...');

// <!-- beforebegin -->
// <element>
//   <!-- afterbegin -->
//   <child>Text</child>
//   <!-- beforeend -->
// </element>
// <!-- afterend -->
    var insert_str = '<div class="balls"><div></div><div></div><div></div></div>';
    target.insertAdjacentHTML('afterbegin', insert_str);
}