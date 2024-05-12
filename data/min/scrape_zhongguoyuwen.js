var wordList = [];
for (let i=1; i <=200; i++){
for (let j=1; j<=5; j++){

const textPath="div.table___3lkbQ:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > table:nth-child(1) > tbody:nth-child(3) > tr:nth-child("+ j +") > td:nth-child(7)"

wordList.push(document.querySelector(textPath).innerHTML);
}
if (i < 4) {
document.querySelector("div.table___3lkbQ:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > ul:nth-child(2) > li:nth-child(9) > button:nth-child(1)").click() 
continue;
}

if (i == 4) {
document.querySelector('li.ant-pagination-next:nth-child(10) > button:nth-child(1)').click()	
continue;
}
if (i < 197) {
document.querySelector('li.ant-pagination-next:nth-child(11) > button:nth-child(1)').click() 
continue;
}
if (i == 197) {
document.querySelector('li.ant-pagination-next:nth-child(10) > button:nth-child(1)').click()
continue;
}
if (i <= 200) {
document.querySelector('div.table___3lkbQ:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > ul:nth-child(2) > li:nth-child(9) > button:nth-child(1)').click()	
continue;
}
}

function exportToCsv(filename, rows) {
    var processRow = function (row) {
        var finalVal = '';
        for (var j = 0; j < row.length; j++) {
            var innerValue = row[j] === null ? '' : row[j].toString();
            if (row[j] instanceof Date) {
                innerValue = row[j].toLocaleString();
            };
            var result = innerValue.replace(/"/g, '""');
            if (result.search(/("|,|\n)/g) >= 0)
                result = '"' + result + '"';
            if (j > 0)
                finalVal += ',';
            finalVal += result;
        }
        return finalVal + '\n';
    };

    var csvFile = '';
    for (var i = 0; i < rows.length; i++) {
        csvFile += processRow(rows[i]);
    }

    var blob = new Blob([csvFile], { type: 'text/csv;charset=utf-8;' });
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        var link = document.createElement("a");
        if (link.download !== undefined) {
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
}

exportToCsv('崇明.csv', [wordList]);
