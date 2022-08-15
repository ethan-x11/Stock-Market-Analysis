const axios = require("axios");

const options = {
  method: 'GET',
  url: 'https://indianstockexchange.p.rapidapi.com/index.php',
  params: {id: '{scrip-id}'},
  headers: {
    'X-RapidAPI-Key': 'f4622008b1mshd7c049bb45247b4p177becjsn35800441b37f',
    'X-RapidAPI-Host': 'indianstockexchange.p.rapidapi.com'
  }
};

axios.request(options).then(function (response) {
	console.log(response.data);
}).catch(function (error) {
	console.error(error);
});