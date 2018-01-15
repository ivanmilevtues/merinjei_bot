function extractIdFromURL(url) {
    // split the sting and take the name of the page
    var result = "";

    result = url.split("/")[3];
    return result;
}

$('#hatespeech-check-btn').click(function () {
    var pageUrl = $('#url-input').val();
    var pageId = extractIdFromURL(pageUrl);

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
    $.ajax({
        type: "POST",
        url: "/classify/scan_for_hatespeech/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "pageId": pageId
        },
        success: function(data) {
            $('#comments-container').append(data);
        }
    });
});