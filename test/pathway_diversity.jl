using ADRIA

@testset "option_similarity returns correct values" begin
    @testset "for standard domain" begin
        dom = ADRIA.load_domain(TEST_DOMAIN_PATH)
        seleceted_locations_1 = dom.loc_data[4:8, :]
        seleceted_locations_2 = dom.loc_data[6:end, :]

        result = ADRIA.analysis.option_similarity(
            seleceted_locations_1, seleceted_locations_2
        )
        @test isapprox(result, 0.99, atol=0.01)
    end

    @testset "for ReefMod Engine domain" begin
        dom = ADRIA.load_domain(ADRIA.RMEDomain, TEST_REEFMOD_ENGINE_DOMAIN_PATH, "45")
        seleceted_locations_1 = dom.loc_data[5:15, :]
        seleceted_locations_2 = dom.loc_data[10:end, :]

        result = ADRIA.analysis.option_similarity(
            seleceted_locations_1, seleceted_locations_2
        )
        @test isapprox(result, 0.98, atol=0.01)
    end
end

@testset "cost_index return correct values" begin
    dom = ADRIA.load_domain(TEST_DOMAIN_PATH, 45)
    ports = ADRIA.analysis._ports()

    @testset "when list only dataframe and no optional argument is passed" begin
        cost_index = ADRIA.analysis.cost_index(dom.loc_data, ports)
        @test isapprox(cost_index, 0.34, atol=0.01)
    end

    @testset "when new weight is passed" begin
        cost_index = ADRIA.analysis.cost_index(dom.loc_data, ports; weight=0.4)
        @test isapprox(cost_index, 0.23, atol=0.01)
    end

    @testset "when new normalization is passed" begin
        cost_index = ADRIA.analysis.cost_index(dom.loc_data, ports;
            max_distance_port=500000.0, max_dispersion=40000.0)
        @test isapprox(cost_index, 0.25, atol=0.01)
    end
end

@testset "_distance_port return correct values" begin
    dom = ADRIA.load_domain(TEST_DOMAIN_PATH, 45)
    ports = ADRIA.analysis._ports()

    distance_port = ADRIA.analysis._distance_port(dom.loc_data, ports)
    @test isapprox(distance_port, 142084.75, atol=0.01)

    selected_ports = [:cairns, :townsville, :gladstone]
    filtred_ports = ports[ports.name .∈ [selected_ports], :]
    distance_port = ADRIA.analysis._distance_port(dom.loc_data, filtred_ports)
    @test isapprox(distance_port, 230851.47, atol=0.01)
end

@testset "_dispersion return correct values" begin
    dom = ADRIA.load_domain(TEST_DOMAIN_PATH, 45)

    dispersion = ADRIA.analysis._dispersion(dom.loc_data)
    @test isapprox(dispersion, 8593.79, atol=0.01)
end
