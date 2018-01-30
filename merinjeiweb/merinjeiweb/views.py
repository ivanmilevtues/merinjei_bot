from django.http import HttpResponseRedirect


def redirect_to_login(request):
    return HttpResponseRedirect('/accounts/login')
