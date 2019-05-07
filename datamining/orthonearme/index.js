
const puppeteer = require('puppeteer');
const fs = require('fs');
// comments thunbs-up / thumbs-down
//Array.from(document.querySelectorAll('div.feedback--list-container div.pure-g div.feedback__body i')).map(label=>label.className)
// feedback--list
// div.feedback--list-container div[data-qa-id="feedback_item"]
async function retriveComment(page)

async function fetch(index,page,TotalCards=[])
{

await page.goto(`https://www.practo.com/bangalore/orthopedist?page=${index}`);

const doctors = await page.evaluate( ()=>{
   
const cards =  Array.from(
       document.querySelectorAll('div[data-qa-id="doctor_card_enhanced"]')).
         map(card => ({
            name : card.querySelector('a h2').textContent,
            recommendation:parseInt(
                (card.querySelector('span[data-qa-id="doctor_recommendation"]'))?
                 card.querySelector('span[data-qa-id="doctor_recommendation"]').textContent :
                 '0'
                ,10),
            url:card.querySelector('div.c-card-info a').href,
            locality:card.querySelector('div.c-card__locality a').textContent,
            hospital:card.querySelector('div.c-card-info__item a[data-qa-id="doctor_clinic_name"]').textContent,
            gmaps:card.querySelector('div.c-card-info__item a[data-qa-id="doctor_clinic_name"]').href
                     }));
    return cards;
});

console.log(`https://www.practo.com/bangalore/orthopedist?page=${index}`,doctors.length);

if(doctors.length < 1)
{
await page.close();
return TotalCards;
}
else{
    TotalCards = TotalCards.concat(doctors);
    return await fetch(index+1,page,TotalCards)
}

}


function writetoFile(obj)
{
fs.writeFile('./json/data.json',
            JSON.stringify(obj,null,2),
            error => error ? 
            console.log('Error ',error) : 
            console.log('Written Successfully')
            );
}


void (async ()=>{
const browser = await puppeteer.launch();

const page =  await browser.newPage();

const cards = await fetch(1,page);

const object = {'bangalore':cards};

writetoFile(object);

await browser.close();
}
)()