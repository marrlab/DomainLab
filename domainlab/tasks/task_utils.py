"""
convert ids to a list of domain names
"""


def parse_domain_id(list_domain_id, list_domains):
    """
    Convert ids to a list of domain names.
    :param list_domain_id: domain id or ids provided as an int or str,
    or a list of int or str.
    :param list_domains: list of available domains
    :return: list of domain names
    """
    if not isinstance(list_domain_id, list):
        list_domain_id = [list_domain_id]
    list_domains_subset = []
    for ele in list_domain_id:
        if isinstance(ele, int):
            list_domains_subset.append(list_domains[ele])
        elif isinstance(ele, str):
            if ele.isdigit():
                list_domains_subset.append(list_domains[int(ele)])
            else:
                list_domains_subset.append(ele)
        else:
            raise RuntimeError("domain ids should be either int or str")
    return list_domains_subset
